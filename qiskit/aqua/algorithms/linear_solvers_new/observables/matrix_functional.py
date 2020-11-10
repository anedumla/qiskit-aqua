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
from typing import Union, Optional
import numpy as np

from .linear_system_observable import LinearSystemObservable
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend, Backend
from qiskit.aqua import QuantumInstance


class MatrixFunctional(LinearSystemObservable):
    """A class for the matrix functional of the vector solution to the linear systems."""

    def __init__(self, matrix: np.ndarray, tolerance: Optional[float] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None) \
            -> None:
        """
        Args:
            matrix: The matrix to compute the functional.
            tolerance: error tolerance.
                Defaults to ``1e-2``.
            quantum_instance: Quantum Instance or Backend
        """
        super().__init__(tolerance, quantum_instance)

        self._matrix = matrix
        self._tolerance = tolerance if tolerance is not None else 1e-2
        self._quantum_instance = quantum_instance

    def evaluate(self, solution: Union[np.ndarray, QuantumCircuit]) -> float:
        r"""Evaluates the matrix functional of the vector solution to the linear systems.

        Args:
            solution: The solution to the system, i.e. x in Ax=b. If the input is
                a circuit, it should prepare a quantum state representing x.

        Returns:
            The value :math:`<x|M|x>`.

        Raises:
            TODO
        """
        #TODO: implement
        raise NotImplementedError
