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

"""An abstract class for linear systems solvers in Qiskit's aqua module."""
from abc import ABC, abstractmethod
from typing import Union, Optional, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend, Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import PauliSumOp


class LinearSystemObservable(ABC):
    """An abstract class for linear system observables in Qiskit's aqua module."""
    #TODO: move outside linear_solvers
    #TODO: add default quantum instance
    #TODO: add classical method from numpy array

    def __init__(self, tolerance: Optional[float] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None) \
            -> None:
        """
        Args:
            tolerance: error tolerance.
                Defaults to ``1e-2``.
            quantum_instance: Quantum Instance or Backend
        Raises:
            ValueError: Invalid input
        """
        self._tolerance = tolerance if tolerance is not None else 1e-2
        # self._quantum_instance = quantum_instance

    @abstractmethod
    def observable(self, num_qubits) -> PauliSumOp:
        """The observable operator.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a sum of Pauli strings.

        Raises:
            TODO
        """
        raise NotImplementedError

    @abstractmethod
    def post_rotation(self, num_qubits: int) -> QuantumCircuit:
        """The observable circuit.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a QuantumCircuit.

        Raises:
            TODO
        """
        raise NotImplementedError

    @abstractmethod
    def post_processing(self, solution: Union[float, List[float]],
                        num_qubits: Optional[int] = None,
                        constant: Optional[Union[float, List[float]]] = 1) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The probability calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            constant: If known, scaling of the solution.

        Returns:
            The value of the observable.

        Raises:
            TODO
        """
        raise NotImplementedError

    @abstractmethod
    def numpy_solver(self, solution: np.array) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The solution to the system as a numpy array.

        Returns:
            The value of the observable.

        Raises:
            TODO
        """
        raise NotImplementedError
