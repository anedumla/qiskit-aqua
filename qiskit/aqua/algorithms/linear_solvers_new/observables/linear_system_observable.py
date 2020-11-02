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
from typing import Union
import numpy as np

from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit


class LinearSystemObservable(ABC):
    """An abstract class for linear system observables in Qiskit's aqua module."""

    @abstractmethod
    def evaluate(self, solution: Union[np.ndarray, BlueprintCircuit]) -> float:
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
