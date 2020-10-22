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

from typing import Optional, Union
import numpy as np
import logging

from .linear_systems_solver import LinearSystemsSolver, LinearSystemsResult
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.providers import BaseBackend
from qiskit.aqua.operators import MatrixTrotterEvolution
from qiskit.aqua.operators.primitive_ops import MatrixOp
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.aqua.components.initial_states import Custom
from inverse_chebyshev import InverseChebyshev

# logger = logging.getLogger(__name__)

class HHL(LinearSystemsSolver):
    def __init__(self,
                 epsilon: float = 1e-2,
                 negative_eigenvalues: bool = True,
                 matrix_circuit: Optional[Union[BlueprintCircuit, QuantumCircuit]] = None,
                 vector_circuit: Optional[Union[BlueprintCircuit, QuantumCircuit]] = None,
                 inverse_circuit: Optional[BlueprintCircuit] = None,
                 function_x: Optional[Union[str, QuantumCircuit]] = None,
                 quantum_instance: Optional[Union[BaseBackend, QuantumInstance]] = None,
                 pec: QuantumCircuit = None) -> None:

        """
        Args:
         matrix: The system matrix A in Ax = b.
         vector:  The rhs vector b in Ax = b.
         epsilon : Error tolerance.
         negative_eigenvalues: A might have negative eigenvalues -- TODO: remove this, algo figures out on its own
         matrix_circuit: Efficient circuit implementing A.
         vector_circuit: Efficient circuit implementing b. TODO: specific name for work qubits
         inverse_circuit: Efficient circuit implementing 1/x. TODO: implement
         function_x: Does post processing on |x> (name tbd).
         quantum_instance: Quantum Instance or Backend.
         pec: Custom circuit for phase estimation with neg. eigenvalues, no use case now, but maybe in future

        .. note::

        Reference Circuit:

        """
        super().__init__(quantum_instance)
        # TODO: better error catches, e.g. np.array type, input could be just vectors
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square!")
        if matrix.shape[0] != len(vector):
            raise ValueError("Input vector dimension does not match input "
                             "matrix dimension!")
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Input matrix must be hermitian!")
        if np.log2(matrix.shape[0]) % 1 != 0:
            raise ValueError("Input matrix dimension must be 2^n!")
        print('test')
        self._matrix = matrix
        self._vector = vector
        self._epsilon = epsilon
        # Tolerance for the different parts of the algorithm
        self._epsilonR = epsilon  # conditioned rotation TODO: change
        self._epsilonS = epsilon  # state preparation TODO: change
        self._epsilonA = epsilon  # hamiltonian simulation TODO: change
        self._nb = np.log2(matrix.shape[0])  # number of qubits for the solution register
        self._nl = self._nb  # TODO:number of qubits for the eigenvalue register
        self._na = self._nb  # TODO:number of ancilla qubits
        self._evo_time = 1  # TODO

        # State preparation circuit - default is qiskit
        if vector_circuit:
            # TODO: if vector_circuit.num_qubits != self._nb: (careful with work qubits)
            # TODO: check if more than one register -> one has to be named work qubits
            # TODO: polynomial, then need epsilon too
            self._vector_circuit = Custom(vector_circuit.num_qubits, circuit=vector_circuit)
        else:
            self._vector_circuit = Custom(self._nb, state_vector=vector)

        # Hamiltonian simulation circuit - default is Trotterization
        # TODO: check if all resize and truncate methods are necessary to have. Matrix_circuit must have _evo_time parameter, control and power methods
        if matrix_circuit:
            self._matrix_circuit = matrix_circuit(self._evo_time, matrix=matrix)
        else:
            # Default is a Trotterization of the matrix TODO:depends on implementation
            evolved_op = MatrixTrotterEvolution(trotter_mode='suzuki', reps=2).convert(
                (self._evo_time * MatrixOp(matrix)).exp_i())

        # Conditioned rotation circuit - inverse is Chebyshev
        if inverse_circuit:
            self._inverse_circuit = inverse_circuit
        else:
            self._inverse_circuit = InverseChebyshev(self.num_state_qubits, self._epsilon, self._constant, self._kappa)

        # Observable gates
        if function_x:
            self._function_x = function_x
        else:
            self._function_x = None

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

    def construct_circuit(self):
        """Construct the HHL circuit.

            Args:

            Returns:
                QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """
        # Initialise the quantum registers
        qb = QuantumRegister(self._nb)  # right hand side and solution
        ql = QuantumRegister(self._nl)  # eigenvalue evaluation qubits
        qa = QuantumRegister(self._na)  # ancilla qubits
        qf = QuantumRegister(2)  # flag qubits

        # Initialise the classical register
        cr = ClassicalRegister(self._nb + self._nl + self._na)

        # Create a Quantum Circuit
        qc = QuantumCircuit(qb, ql, qa, qf, cr)

        # State preparation - TODO: check if work qubits
        self._vector_circuit.construct_circuit("circuit", qb)
        # QPE
        qc.append(PhaseEstimation(self._nl, self._matrix_circuit), qb[:] + ql[:])
        # Conditioned rotation
        self._inverse_circuit._build(qc, qb, qf[1], qa)
        # QPE inverse
        qc.append(PhaseEstimation(self._nl, self._matrix_circuit.inverse), qb[:] + ql[:])  # TODO: sort out inverse
        # Observable gates
        if self._function_x:
            self._function_x.construct_circuit("circuit", qb)

    def solve(self, matrix: Union[np.ndarray, BlueprintCircuit],
              vector: Union[np.ndarray, BlueprintCircuit]) -> 'LinearSystemsResult':
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.

        Returns:
            The result of the linear system.
        """
        return LinearSystemsResult()


"""Test:
State preparation with vector
State preparation with circuit
"""