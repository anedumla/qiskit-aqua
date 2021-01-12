# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""TODO: The HHL algorithm."""

from typing import Optional, Union, List
import numpy as np
import logging

from .linear_solver import LinearSolver, LinearSolverResult
from .observables.linear_system_observable import LinearSystemObservable
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, Aer, execute
from qiskit.circuit.library import PhaseEstimation
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.aqua.components.reciprocals import LookupRotation
from qiskit.aqua.algorithms.linear_solvers_new.exact_inverse import ExactInverse
# logger = logging.getLogger(__name__)

# TODO: allow a different eigenvalue circuit
# TODO: need negative eigenvalues? try that algorithm figures out on its own
class HHL(LinearSolver):
    def __init__(self,
                 epsilon: float = 1e-2,
                 quantum_instance: Optional[Union[BaseBackend, QuantumInstance]] = None,
                 pec: QuantumCircuit = None) -> None:

        """
        Args:
         epsilon : Error tolerance.
         quantum_instance: Quantum Instance or Backend.
         pec: Custom circuit for phase estimation with neg. eigenvalues, no use case now, but maybe in future

        .. note::

        References:

        [1]: Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2020).
             Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.
             `arXiv:2009.04484 <http://arxiv.org/abs/2009.04484>`_

        """
        super().__init__()

        self._epsilon = epsilon
        # Tolerance for the different parts of the algorithm as per [1]
        self._epsilonR = epsilon / 3  # conditioned rotation TODO: change
        self._epsilonS = epsilon / 3  # state preparation TODO: change
        self._epsilonA = epsilon / 6  # hamiltonian simulation TODO: change

        # Number of qubits for the different registers
        self._nb = None  # number of qubits for the solution register
        self._nl = None  # TODO:number of qubits for the eigenvalue register
        self._na = None  # TODO:number of ancilla qubits
        # Time of the evolution. Once the matrix is specified,
        # it can be updated 2 * np.pi / lambda_max
        self._evo_time = 2 * np.pi
        self._lambda_max = None
        self._lambda_min = None
        self._kappa = 1 # Set default condition number to 1

        # Circuits for the different blocks of the algorithm
        self._vector_circuit = None
        self._matrix_circuit = None
        self._inverse_circuit = None
        self._function_x = None

        # Temporary fix - TODO update how to decide if exact inverse
        self._exact_inverse = True

    @property
    def _num_qubits(self) -> int:
        """TODO: implement. Return the number of qubits needed in the circuit.

        Returns:
            The total number of qubits.
        """
        if self.a_factory is None:  # if A factory is not set, no qubits are specified
            return 0

        num_ancillas = self.q_factory.required_ancillas()
        num_qubits = self.a_factory.num_target_qubits + num_ancillas

        return num_qubits

    def get_delta(self, nl, lambda_min, lambda_max):
        formatstr = "#0" + str(nl + 2) + "b"
        lambda_min_tilde = np.abs(lambda_min * (2 ** nl - 1) / lambda_max)
        binstr = format(int(lambda_min_tilde), formatstr)[2::]
        lamb_min_rep = 0
        for i in range(0, len(binstr)):
            lamb_min_rep += int(binstr[i]) / (2 ** (i + 1))
        return lamb_min_rep


    def calculate_norm(self, qc: QuantumCircuit) -> float:
        # Measure
        backend = Aer.get_backend('statevector_simulator')
        # Create a Quantum Program for execution
        job = execute(qc, backend)
        # and store the result
        result = job.result()
        # Remember its an...a0qn..q0
        svect = result.get_statevector(qc)

        lastq = self._nb + self._na + self._nl + 2
        nflags = 2
        formatstr = "{0:0" + str(lastq) + "b}"
        inv = [0] * (2 ** self._nb)
        totalnorm = 0
        for i in range(0, len(svect)):
            if int(formatstr.format(i)[0]) == 0 and int(formatstr.format(i)[1]) == 1 and int(
                    formatstr.format(i)[2]) == 1:
                if np.linalg.norm(svect[i]) != 0.0:
                    if formatstr.format(i)[nflags:nflags + self._na + self._nl] == '0' * (self._nl + self._na):
                        totalnorm += np.real(svect[i] * np.conjugate(svect[i]))
                        current = int(formatstr.format(i)[lastq - self._nb:lastq], 2)
                        inv[current] += svect[i]
        return np.sqrt(totalnorm * (2 ** self._nb)) / self._lambda_min

    def construct_circuit(self) -> QuantumCircuit:
        """Construct the HHL circuit.

            Args:

            Returns:
                QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """
        # Initialise the quantum registers
        qb = QuantumRegister(self._nb)  # right hand side and solution
        ql = QuantumRegister(self._nl)  # eigenvalue evaluation qubits
        if self._na > 0:
            qa = AncillaRegister(self._na)  # ancilla qubits
        qf = QuantumRegister(1)
        # qf = QuantumRegister(2)  # flag qubits

        # Initialise the classical register
        # cr = ClassicalRegister(self._nb + self._nl + self._na)

        # Create a Quantum Circuit
        # qc = QuantumCircuit(qb, ql, qa, qf, cr)
        if self._na >0:
            qc = QuantumCircuit(qb, ql, qa, qf)
        else:
            qc = QuantumCircuit(qb, ql, qf)

        # State preparation - TODO: check if work qubits
        qc.append(self._vector_circuit, qb[:])
        # QPE
        phase_estimation = PhaseEstimation(self._nl, self._matrix_circuit)
        if self._na >0:
            qc.append(phase_estimation, ql[:] + qb[:] + qa[:self._matrix_circuit.num_ancillas])
        else:
            qc.append(phase_estimation, ql[:] + qb[:])
        # Conditioned rotation
        if self._exact_inverse:
            qc.append(self._inverse_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(self._inverse_circuit.to_instruction(), ql[:] + [qf[0]] +
                      qa[:self._inverse_circuit.num_ancillas])
        # QPE inverse
        if self._na > 0:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:] + qa[:self._matrix_circuit.num_ancillas])
        else:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:])
        # Observable gates
        if self._function_x:
            self._function_x.construct_circuit("circuit", qb)
        return qc

    #TODO: neg eigenvalues, general matrix, observable:List[BaseOperator],update notebook
    """
    solve(matrix,vector,
          observable:Optional[Union[BaseOperator,List[BaseOperator]]],
          post_rotation:Optional[Union[QuantumCircuit,List[QuantumCircuit]]],
          post_processing:Optional[Callable[Union[float,List[float]],Union[float,List[float]]]])
    post_rotation: circuit, can contain none
        if none->
    """
    def solve(self, matrix: Union[np.ndarray, QuantumCircuit],
              vector: Union[np.ndarray, QuantumCircuit],
              observable: Optional[Union[LinearSystemObservable, List[LinearSystemObservable]]]
              = None) -> LinearSolverResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Information to be extracted from the solution.
                Default is `EuclideanNorm`

        Returns:
            The result of the linear system.
        """
        # TODO: better error catches, e.g. np.array type, input could be just vectors
        # State preparation circuit - default is qiskit
        if isinstance(vector, QuantumCircuit):
            # TODO: check if more than one register -> one has to be named work qubits
            # TODO: polynomial, then need epsilon too
            self._nb = vector.num_qubits
            self._vector_circuit = vector
        elif isinstance(vector, np.ndarray):
            self._nb = int(np.log2(len(vector)))
            self._vector_circuit = QuantumCircuit(self._nb).initialize(vector /
                                                                       np.linalg.norm(vector),
                                                                       list(range(self._nb)))
        else:
            raise ValueError("Input vector type must be either QuantumCircuit or numpy ndarray.")

        # Hamiltonian simulation circuit - default is Trotterization
        # TODO: check if all resize and truncate methods are necessary to have.
        #  Matrix_circuit must have _evo_time parameter, control and power methods
        if isinstance(matrix, QuantumCircuit):
            self._matrix_circuit = matrix
            matrix_array = matrix.matrix()
        elif isinstance(matrix, np.ndarray):
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2 ** self._vector_circuit.num_qubits:
                raise ValueError("Input vector dimension does not match input "
                                 "matrix dimension!")
            matrix_array = matrix

            # Default is a Trotterization of the matrix TODO:depends on implementation
            # evolved_op = MatrixTrotterEvolution(trotter_mode='suzuki', reps=2).convert(
            #     (self._evo_time * MatrixOp(matrix)).exp_i())
        else:
            raise ValueError("Input matrix type must be either QuantumCircuit or numpy ndarray.")


        self._lambda_max = max(np.abs(np.linalg.eigvals(matrix_array)))
        self._lambda_min = min(np.abs(np.linalg.eigvals(matrix_array)))
        self._kappa = np.linalg.cond(matrix_array)
        # Update the number of qubits required to represent the eigenvalues
        # self._nl = min(self._nb + 1, 3 * (int(np.log2(2 * (2 * (self._kappa ** 2) - self._epsilonR) /
        #                                               self._epsilonR + 1) + 1)))
        self._nl = max(self._nb +1, int(np.log2(self._lambda_max/self._lambda_min))+1)

        # Constant from the representation of eigenvalues
        delta = self.get_delta(self._nl, self._lambda_min, self._lambda_max)

        # Update evolution time - TODO: leave it as optional since not doable for all matrices
        #  without killing the speedup
        self._evo_time = 2 * np.pi * delta / self._lambda_min
        self._matrix_circuit.set_simulation_params(self._evo_time, self._epsilonA)

        if self._exact_inverse:
            self._inverse_circuit = ExactInverse(self._nl, delta)
            # Update number of ancilla qubits
            self._na = self._matrix_circuit.num_ancillas

        else:
            # Calculate breakpoints for the inverse approximation
            num_values = 2 ** self._nl
            constant = delta
            # constant = num_values * self._evo_time * self._lambda_min / 2 / np.pi
            a = int(round(num_values ** (2 / 3)))  # pylint: disable=invalid-name

            # Calculate the degree of the polynomial and the number of intervals
            r = 2 * constant / a + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            degree = min(self._nb, int(np.log(1 + (16.23 * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2) *
                                                   self._kappa * (2 * self._kappa - self._epsilonR)) /
                                              self._epsilonR)))
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))

            # Calculate breakpoints and polynomials
            breakpoints = []
            for i in range(0, num_intervals):
                # Add the breakpoint to the list
                breakpoints.append(a * (5 ** i))

                # Define the right breakpoint of the interval
                if i == num_intervals - 1:
                    breakpoints.append(num_values - 1)

            inverse_circuit = PiecewiseChebyshev(lambda x: np.arcsin(constant / x), degree, breakpoints, self._nl)
            self._inverse_circuit = inverse_circuit.to_instruction()
            self._inverse_circuit._build()
            self._na = max(self._matrix_circuit.num_ancillas, inverse_circuit.num_ancillas)

        solution = LinearSolverResult()
        solution.state = self.construct_circuit()
        solution.euclidean_norm = self.calculate_norm(solution.state)
        return solution

#TODO set the euclidean_norm property
"""Test:
State preparation with vector
State preparation with circuit
"""
