from typing import Optional

import numpy as np
from scipy.sparse import diags

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister, Qubit

class Tridiagonal(QuantumCircuit):
    """Temporal class to test the new hhl algorithm"""

    def __init__(self, num_state_qubits: int, main_diag: float, off_diag: float,
                 tolerance: Optional[float] = None, time: Optional[float] = None,
                 trotter: Optional[int] = 1, name: str = 'tridi') -> None:

        qr_state = QuantumRegister(num_state_qubits)
        if num_state_qubits > 1:
            qr_ancilla = AncillaRegister(max(1, num_state_qubits - 1))
            super().__init__(qr_state, qr_ancilla, name=name)
        else:
            super().__init__(qr_state, name=name)

        self._num_state_qubits = None

        self._main_diag = main_diag
        self._off_diag = off_diag
        self._tolerance = tolerance if tolerance is not None else 1e-2
        self._time = time if time is not None else 1
        self._trotter = trotter

        self._num_state_qubits = num_state_qubits

    def set_simulation_params(self, time: float, tolerance: float):
        self.tolerance(tolerance)
        self.time(time)

    def tolerance(self, tolerance: float):
        self._tolerance = tolerance

    def time(self, time: float):
        self._time = time
        # Update the number of trotter steps. Max 7 for now, upper bounds too loose.
        self._trotter = min(self._num_state_qubits + 1,int(np.ceil(np.sqrt(((time * np.abs(self._off_diag)) ** 3) / 2 /
                                                  self._tolerance))))

    def matrix(self) -> np.ndarray:
        """Return the matrix"""
        matrix = diags([self._off_diag, self._main_diag, self._off_diag], [-1, 0, 1],
                       shape=(2 ** self._num_state_qubits, 2 ** self._num_state_qubits)).toarray()
        return matrix

    def _cn_gate(self, qc: QuantumCircuit, controls: QuantumRegister, qr_a: AncillaRegister,
                 phi: float, ulambda: float, theta: float, tgt: Qubit):
        """Apply an n-controlled gate.

        Args:
            controls: list of controlling qubits
            qr_a: ancilla register
            phi: argument for a general qiskit u gate
            ulambda: argument for a general qiskit u gate
            theta: argument for a general qiskit u gate
            tgt: target qubit
        """
        # The first Toffoli
        qc.ccx(controls[0], controls[1], qr_a[0])
        for i in range(2, len(controls)):
            qc.ccx(controls[i], qr_a[i - 2], qr_a[i - 1])
        # Now apply the 1-controlled version of the gate with control the last ancilla bit
        qc.cu(theta, phi, ulambda, 0, qr_a[len(controls) - 2], tgt)

        # Uncompute ancillae
        for i in range(len(controls) - 1, 1, -1):
            qc.ccx(controls[i], qr_a[i - 2], qr_a[i - 1])
        qc.ccx(controls[0], controls[1], qr_a[0])
        return qc

    # Controlled version of the circuit for the main diagonal
    def _build_main_controlled(self, qc: QuantumCircuit, q_control: Qubit, params: Optional[float] = 1):
        """Controlled circuit for the matrix consisting of entries in the main diagonal.

        Args:
            q_control: The control qubit.
            params: Argument for the rotation.
        """
        qc.p(params, q_control)
        return qc

    # Controlled version of the circuit for the main diagonal
    def _build_off_diag_controlled(self, qc: QuantumCircuit, q_control: Qubit, qr: QuantumRegister,
                                   qr_anc: Optional[AncillaRegister] = None,
                                   params: Optional[float] = 1) -> QuantumCircuit:
        """Controlled circuit for the matrix consisting of entries in the off diagonals.

        Args:
            qc: The quantum circuit.
            q_control: The control qubit.
            qr: The quantum register where the circuit is built.
            qr_anc: The quantum register containing the ancilla qubits.
            params: Argument for the rotation.
        """
        # Gates for H2 with t
        qc.cu(-2 * params, 3 * np.pi / 2, np.pi / 2, 0, q_control, qr[0])

        # Gates for H3
        for i in range(0, self._num_state_qubits - 1):
            q_controls = []
            q_controls.append(q_control)
            qc.cx(qr[i], qr[i + 1])
            q_controls.append(qr[i + 1])

            # Now we want controlled by 0
            qc.x(qr[i])
            for j in range(i, 0, -1):
                qc.cx(qr[i], qr[j - 1])
                q_controls.append(qr[j - 1])
            qc.x(qr[i])

            # Multicontrolled x rotation
            if len(q_controls) > 1:
                self._cn_gate(qc, q_controls, qr_anc, 3 * np.pi / 2, np.pi / 2, -2 * params, qr[i])
            else:
                qc.cu(-2 * params, 3 * np.pi / 2, np.pi / 2, 0, q_controls[0], qr[i])

            # Uncompute
            qc.x(qr[i])
            for j in range(0, i):
                qc.cx(qr[i], qr[j])
            qc.x(qr[i])
            qc.cx(qr[i], qr[i + 1])

        return qc

    def inverse(self):
        self._time = - self._time

    def power(self, power: int):
        """Build powers of the circuit.

        Args:
            power: The exponent.
        """
        qc_raw = QuantumCircuit(self._num_state_qubits)

        def control():
            qr_state = QuantumRegister(self._num_state_qubits + 1)
            if self._num_state_qubits > 1:
                qr_ancilla = AncillaRegister(max(1, self._num_state_qubits - 1))
                qc = QuantumCircuit(qr_state, qr_ancilla)
            else:
                qc = QuantumCircuit(qr_state)
                qr_ancilla = None
            # Control will be qr[0]
            q_control = qr_state[0]
            qr = qr_state[1:]
            # Since A1 commutes, one application with time t*2^{j} to the last qubit is enough
            self._build_main_controlled(qc, q_control, self._main_diag * self._time * power)

            # Update trotter step to compensate the error
            trotter_new = int(np.ceil(np.sqrt(power) * self._trotter))

            # exp(iA2t/2m)
            qc.u(self._off_diag * self._time * power / trotter_new, 3 * np.pi / 2, np.pi / 2, qr[0])
            # for _ in range(power):
            for _ in range(0, trotter_new):
                self._build_off_diag_controlled(qc, q_control, qr, qr_ancilla,
                                                self._time * self._off_diag * power / trotter_new)
            # exp(-iA2t/2m)
            qc.u(-self._off_diag * self._time * power / trotter_new, 3 * np.pi / 2, np.pi / 2, qr[0])
            return qc

        qc_raw.control = control
        return qc_raw

    # @property
    # def tolerance(self) -> float:
    #     """The error tolerance of the approximation to Hamiltonian simulation.
    #
    #     Returns:
    #         The error tolerance.
    #     """
    #     return self._tolerance
    #
    # @tolerance.setter
    # def tolerance(self, tolerance: Optional[float]):
    #     """Set the error tolerance.
    #
    #     Args:
    #         tolerance: The new error tolerance."""
    #     self._tolerance = tolerance
    #
    # @property
    # def time(self) -> float:
    #     """The time for the Hamiltonian evolution.
    #
    #     Returns:
    #         The time parameter.
    #     """
    #     return self._time
    #
    # @time.setter
    # def time(self, time: Optional[float]):
    #     """Set the time for the Hamiltonian evolution.
    #
    #     Args:
    #         time: The new time parameter."""
    #     self._time = time
    #
    # @property
    # def trotter(self) -> int:
    #     """The number of trotter steps.
    #
    #     Returns:
    #         The number of trotter steps.
    #     """
    #     return self._trotter
    #
    # @trotter.setter
    # def trotter(self, trotter: Optional[int]):
    #     """Set the number of trotter steps.
    #
    #     Args:
    #         trotter: The new number of trotter steps."""
    #     self._trotter = trotter
    #
    # @property
    # def num_state_qubits(self) -> int:
    #     r"""The number of state qubits representing the state :math:`|x\rangle`.
    #
    #     Returns:
    #         The number of state qubits.
    #     """
    #     return self._num_state_qubits
    #
    # @num_state_qubits.setter
    # def num_state_qubits(self, num_state_qubits: Optional[int]) -> None:
    #     """Set the number of state qubits.
    #
    #     Note that this may change the underlying quantum register, if the number of state qubits
    #     changes.
    #
    #     Args:
    #         num_state_qubits: The new number of qubits.
    #     """
    #     if self._num_state_qubits is None or num_state_qubits != self._num_state_qubits:
    #         self._invalidate()
    #         self._num_state_qubits = num_state_qubits
    #
    #         self._reset_registers(num_state_qubits)
    #
    # def _check_configuration(self, raise_on_failure: bool = True) -> bool:
    #     valid = True
    #
    #     if self.num_state_qubits is None:
    #         valid = False
    #         if raise_on_failure:
    #             raise AttributeError('The number of qubits has not been set.')
    #
    #     if self._main_diag is None or self._off_diag:
    #         valid = False
    #         if raise_on_failure:
    #             raise AttributeError('The values of the main and off diagonals must be set.')
    #
    #     return valid
    #
    # def _reset_registers(self, num_state_qubits: Optional[int], controlled: bool = False) -> None:
    #     if num_state_qubits:
    #         qr_state = QuantumRegister(num_state_qubits)
    #         self.qregs = [qr_state]
    #
    #         # Calculate number of ancilla qubits required
    #         if self.num_state_qubits == 1:
    #             num_ancillas = 0
    #         else:
    #             if controlled:
    #                 num_ancillas = max(1, self._num_state_qubits - 1)
    #             else:
    #                 num_ancillas = max(1, self._num_state_qubits - 2)
    #
    #         if num_ancillas > 0:
    #             self._ancillas = []
    #             qr_ancilla = AncillaRegister(num_ancillas)
    #             self.add_register(qr_ancilla)
    #     else:
    #         self.qregs = []



    # # Circuit for the main diagonal matrix
    # def _build_main(self, q_0: Qubit, params: Optional[float] = 1):
    #     """Circuit for the matrix consisting of entries in the main diagonal.
    #
    #     Args:
    #         q_0: The qubit where the rotation is applied.
    #         params: Argument for the rotation.
    #     """
    #     self.x(q_0)
    #     self.u1(params, q_0)
    #     self.x(q_0)
    #     self.u1(params, q_0)



    # # Circuit for the off diagonal matrix
    # def _build_off_diag(self, qc: QuantumCircuit, qr: QuantumRegister,
    #                     qr_anc: Optional[AncillaRegister] = None, m_trotter: Optional[int] = 1,
    #                     params: Optional[float] = 1) -> QuantumCircuit:
    #     """Circuit for the matrix consisting of entries in the off diagonals.
    #
    #     Args:
    #         qr: The quantum register where the circuit is built.
    #         qr_anc: The quantum register containing the ancilla qubits.
    #         m_trotter: The trotter exponent.
    #         params: Argument for the rotation.
    #     """
    #     # Gates for H2
    #     self.u3(-2 * params, -np.pi / 2, np.pi / 2, qr[0])
    #
    #     # Gates for H3
    #     for i in range(0, self.num_state_qubits - 1):
    #         q_controls = []
    #         self.cx(qr[i], qr[i + 1])
    #         q_controls.append(qr[i + 1])
    #
    #         # Now we want controlled by 0
    #         self.x(qr[i])
    #         for j in range(i, 0, -1):
    #             self.cx(qr[i], qr[j - 1])
    #             q_controls.append(qr[j - 1])
    #         self.x(qr[i])
    #
    #         # Multicontrolled x rotation
    #         if len(q_controls) > 1:
    #             self.append(self._cn_gate(q_controls, qr_anc, -np.pi / 2, np.pi / 2,
    #                                       -2 * params, qr[i]), qr_anc[:] + qr[:])
    #         else:
    #             self.cu3(-2 * params, -np.pi / 2, np.pi / 2, q_controls[0], qr[i])
    #         # Uncompute
    #         self.x(qr[i])
    #         for j in range(0, i):
    #             self.cx(qr[i], qr[j])
    #         self.x(qr[i])
    #         self.cx(qr[i], qr[i + 1])
    #
    #     return qc



    # def _build(self):
    #     super()._build()
    #
    #     qr_state = self.qubits[:self.num_state_qubits]
    #     qr_ancilla = self.ancillas
    #
    #     # Since H1 commutes, one application with time t*2^{j} to the last qubit is enough
    #     self._build_main(qr_state[0], self._time * self._main_diag)
    #
    #     # exp(iA2t/2m)
    #     self.u3(self._off_diag * self._time / self._trotter, -np.pi / 2, np.pi / 2, qr_state[0])
    #     for _ in range(0, self._trotter):
    #         self._build_off_diag(qr_state, qr_ancilla, self._trotter,
    #                              self._time * self._off_diag / self._trotter)
    #     # exp(-iA2t/2m)
    #     self.u3(-self._off_diag * self._time / self._trotter, -np.pi / 2, np.pi / 2, qr_state[0])
    #
    # def inverse(self):
    #     super()._build()
    #
    #     qr_state = self.qubits[:self.num_state_qubits]
    #     qr_ancilla = self.ancillas
    #
    #     # Since H1 commutes, one application with time t*2^{j} to the last qubit is enough
    #     self._build_main(qr_state[0], -self._time * self._main_diag)
    #
    #     # exp(iA2t/2m)
    #     self.u3(-self._off_diag * self._time / self._trotter, -np.pi / 2, np.pi / 2, qr_state[0])
    #     for _ in range(0, self._trotter):
    #         self._build_off_diag(qr_state, qr_ancilla, self._trotter,
    #                              -self._time * self._off_diag / self._trotter)
    #     # exp(-iA2t/2m)
    #     self.u3(self._off_diag * self._time / self._trotter, -np.pi / 2, np.pi / 2, qr_state[0])



    # def control(self):
    #     self._reset_registers(self.num_state_qubits, True)
    #     super()._build()
    #     qr_state = self.qubits[:self.num_state_qubits]
    #     qr_ancilla = self.qubits[self.num_state_qubits:]
    #     self.x(qr_state[0])
    #     self.h(qr_state[0])
