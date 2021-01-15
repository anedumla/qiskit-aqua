# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The matrix functional of the vector solution to the linear systems."""
from typing import Union, Optional, List
import numpy as np

from qiskit.aqua.operators import PauliSumOp
from .linear_system_observable import LinearSystemObservable
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend, Backend
from qiskit.aqua import QuantumInstance
from qiskit.opflow import I, Z, Zero, One
from scipy.sparse import diags


class MatrixFunctional(LinearSystemObservable):
    """A class for the matrix functional of the vector solution to the linear systems."""

    def __init__(self, main_diag: float, off_diag: int, tolerance: Optional[float] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None) \
            -> None:
        """
        Args:
            main_diag: The main diagonal of the tridiagonal Toeplitz symmetric matrix to compute
             the functional.
            off_diag: The off diagonal of the tridiagonal Toeplitz symmetric matrix to compute
             the functional.
            tolerance: error tolerance.
                Defaults to ``1e-2``.
            quantum_instance: Quantum Instance or Backend
        """
        super().__init__(tolerance, quantum_instance)

        self._main_diag = main_diag
        self._off_diag = off_diag
        self._tolerance = tolerance if tolerance is not None else 1e-2
        # self._quantum_instance = quantum_instance

    def observable(self, num_qubits: int) -> List[PauliSumOp]:
        """The observable operators.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a list of sums of Pauli strings.

        Raises:
            TODO
        """
        ZeroOp = ((I + Z) / 2)
        OneOp = ((I - Z) / 2)
        observables = []
        # First we measure the norm of x
        observables.append(I ^ num_qubits)
        for i in range(0,num_qubits):
            j = num_qubits - i - 1
            observables.append([(I ^ j) ^ ZeroOp ^ (OneOp ^ i), (I ^ j) ^ OneOp ^ (OneOp ^ i)])
            # observables.append((I ^ j) ^ (ZeroOp - OneOp) ^ (OneOp ^ i))

        return observables

    def post_rotation(self, num_qubits: int) -> List[QuantumCircuit]:
        """The observable circuits.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a list of QuantumCircuits.

        Raises:
            TODO
        """
        qcs = []
        # Again, the first value in the list will correspond to the norm of x
        qcs.append(QuantumCircuit(num_qubits))
        for i in range(0, num_qubits):
            qc = QuantumCircuit(num_qubits)
            for j in range(0, i):
                qc.cx(i, j)
            qc.h(i)
            qcs.append(qc)

        return qcs

    # TODO: can remove the list option or does it have to be as in the abstract method?.
    def post_processing(self, solution: Union[float, List[float]],
                        num_qubits: int,
                        constant: Optional[Union[float, List[float]]] = 1) -> float:
        """Evaluates the matrix functional on the solution to the linear system.

        Args:
            solution: The list of probabilities calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            constant: If known, scaling of the solution.

        Returns:
            The value of the absolute average.

        Raises:
            TODO
        """
        if num_qubits is None:
            raise ValueError("Number of qubits must be defined to calculate the absolute average.")
        # Calculate the value from the off-diagonal elements
        off_val = 0
        for v in solution[1::]:
            # if np.abs(v[0]) >= np.abs(v[1]):
            off_val += (v[0]-v[1]) / (constant ** 2)
            # else:
            #     off_val += (v[1] - v[0]) * (2 ** num_qubits) / (constant ** 2)
        main_val = solution[0] / (constant ** 2)
        return np.real(self._main_diag * main_val + self._off_diag * off_val)

    def numpy_solver(self, solution: np.array) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The solution to the system as a numpy array.

        Returns:
            The value of the observable.

        Raises:
            TODO
        """

        matrix = diags([self._off_diag, self._main_diag, self._off_diag], [-1, 0, 1],
                       shape=(len(solution), len(solution))).toarray()

        return np.dot(solution.transpose(), np.dot(matrix, solution))
