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
from typing import Dict, Union
import numpy as np

from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit

from qiskit.aqua.algorithms import AlgorithmResult


class LinearSystemsSolver(ABC):
    """An abstract class for linear system solvers in Qiskit's aqua module."""

    # @abstractmethod
    # def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
    #     """Checks whether a given problem can be solved with the optimizer implementing this method.
    #
    #     Args:
    #         problem: The optimization problem to check compatibility.
    #
    #     Returns:
    #         Returns the incompatibility message. If the message is empty no issues were found.
    #     """
    #
    # def is_compatible(self, problem: QuadraticProgram) -> bool:
    #     """Checks whether a given problem can be solved with the optimizer implementing this method.
    #
    #     Args:
    #         problem: The optimization problem to check compatibility.
    #
    #     Returns:
    #         Returns True if the problem is compatible, False otherwise.
    #     """
    #     return len(self.get_compatibility_msg(problem)) == 0

    # TODO: add repeat-until-success circuit option
    # TODO: add Observable parameter
    @abstractmethod
    def solve(self, matrix: Union[np.ndarray, BlueprintCircuit],
              vector: Union[np.ndarray, BlueprintCircuit]) -> 'LinearSystemsResult':
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.

        Returns:
            The result of the linear system.

        Raises:
            TODO
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        raise NotImplementedError


class LinearSystemsResult(AlgorithmResult):
    """A base class for linear systems results.

    The linear systems algorithms return an object of the type ``LinearSystemsResult``
    with the information about the solution obtained.

    ``LinearSystemsResult`` allows users to get the value of a variable by specifying an index or
    a name as follows TODO.

    Examples:
        >>> TODO

    Note:
        TODO
    """

    @property
    def solution(self) -> np.ndarray:
        """ return solution """
        return self.get('solution')

    @solution.setter
    def solution(self, value: np.ndarray) -> None:
        """ set solution """
        self.data['solution'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'LinearsolverResult':
        """ create new object from a dictionary """
        return LinearSystemsResult(a_dict)
