# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test NumPy LS solver """

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np

from qiskit import QuantumCircuit
from qiskit.aqua.algorithms.linear_solvers_new.numpy_linear_solver import NumPyLinearSolver


class TestNumPyLSsolver(QiskitAquaTestCase):
    """ Test NumPy LS solver """
    def setUp(self):
        super().setUp()
        self.matrix = [[1, 2], [2, 1]]
        self.vector = [1, 2]

    def test_els(self):
        """ ELS test """
        algo = NumPyLinearSolver()
        solution = algo.solve(self.matrix, self.vector)
        np.testing.assert_array_almost_equal(solution.observable, [1, 0])

        # Test raise error
        with self.assertRaises(ValueError):
            algo.solve(self.matrix, QuantumCircuit(1))
        with self.assertRaises(ValueError):
            algo.solve(QuantumCircuit(1), self.vector)

if __name__ == '__main__':
    unittest.main()
