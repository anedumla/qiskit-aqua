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
from typing import Union, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend, Backend
from qiskit.aqua import QuantumInstance


class LinearSystemObservable(ABC):
    """An abstract class for linear system observables in Qiskit's aqua module."""
    #TODO: move outside linear_solvers
    #TODO: add default quantum instance

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
        self._quantum_instance = quantum_instance

    @abstractmethod
    def evaluate(self, solution: Union[np.ndarray, QuantumCircuit]) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The solution to the system, i.e. x in Ax=b. If the input is
                a circuit, it should prepare a quantum state representing x.

        Returns:
            The value of the observable.

        Raises:
            TODO
        """
        raise NotImplementedError
