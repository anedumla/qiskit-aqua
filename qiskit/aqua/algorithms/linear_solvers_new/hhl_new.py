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
import scipy as sp
import logging

from .linear_solver import LinearSolver, LinearSolverResult
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
        self._kappa = 1 # Set default condition number to 1

        # Circuits for the different blocks of the algorithm
        self._vector_circuit = None
        self._matrix_circuit = None
        self._inverse_circuit = None

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
              vector: Union[np.ndarray, BlueprintCircuit]) -> 'LinearSolverResult':
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.

        Returns:
            The result of the linear system.
        """
        # TODO: better error catches, e.g. np.array type, input could be just vectors
        # State preparation circuit - default is qiskit
        if isinstance(vector, BlueprintCircuit):
            # TODO: check if more than one register -> one has to be named work qubits
            # TODO: polynomial, then need epsilon too
            self._nb = vector.num_qubits
        elif isinstance(vector, np.ndarray):
            self._nb = int(np.log2(len(vector)))
        else:
            raise ValueError("Input vector type must be either BlueprintCircuit or numpy ndarray.")
        self._vector_circuit = Custom(self._nb, state_vector=vector)

        # Hamiltonian simulation circuit - default is Trotterization
        # TODO: check if all resize and truncate methods are necessary to have.
        #  Matrix_circuit must have _evo_time parameter, control and power methods
        if isinstance(matrix, BlueprintCircuit):
            self._matrix_circuit = matrix(self._evo_time, matrix=matrix)
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
            # Update evolution time - TODO: leave it as optional since not doable for all matrices
            #  without killing the speedup
            self._lambda_max = max(np.abs(np.linalg.eigvals(matrix)))
            self._kappa = np.linalg.cond(matrix)
            self._evo_time = 2 * np.pi / self._lambda_max
            # Default is a Trotterization of the matrix TODO:depends on implementation
            evolved_op = MatrixTrotterEvolution(trotter_mode='suzuki', reps=2).convert(
                (self._evo_time * MatrixOp(matrix)).exp_i())
        else:
            raise ValueError("Input matrix type must be either BlueprintCircuit or numpy ndarray.")

        # Update the number of qubits required to represent the eigenvalues
        self._nl = 3 * (int(np.log2(2 * (2 * (self._kappa ** 2) - self._epsilonR) /
                                    self._epsilonR + 1) + 1))
        self._inverse_circuit = InverseChebyshev(self._nl, self._epsilon, self._constant, self._kappa)

        return LinearSolverResult()


"""Test:
State preparation with vector
State preparation with circuit
"""
