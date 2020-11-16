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
"""The Numpy LinearSolver algorithm (classical)."""

from typing import List, Union, Optional
import logging
import numpy as np

from .linear_solver import LinearSolverResult, LinearSolver
from .observables.linear_system_observable import LinearSystemObservable

logger = logging.getLogger(__name__)


class NumPyLinearSolver(LinearSolver):
    """
    The Numpy LinearSolver algorithm (classical).

    This linear system solver computes the exact value of the given observable(s) or the full
    solution vector if no observable is specified.
    """

    def __init__(self) -> None:
        super().__init__()
        self._solution = LinearSolverResult()

    def solve(self, matrix: np.ndarray, vector: np.ndarray,
              observable: Optional[Union[LinearSystemObservable, List[LinearSystemObservable]]]
              = None) -> LinearSolverResult:
        """
        solve the system and compute the observable(s)
        Returns:
            LinearSolverResult
        """
        solution_vector = np.linalg.solve(matrix, vector)
        if observable is not None:
            self._solution.result = observable.evaluate(solution_vector)
        else:
            self._solution.result = solution_vector
        self._solution.euclidean_norm = np.linalg.norm(solution_vector)
        return self._solution
