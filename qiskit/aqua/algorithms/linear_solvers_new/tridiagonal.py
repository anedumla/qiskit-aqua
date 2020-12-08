from typing import Optional

from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit


class Tridiagonal(BlueprintCircuit):
    """Temporal class to test the new hhl algorithm"""

    def __init__(self, tolerance: Optional[float] = None, time: Optional[float] = None,
                 num_state_qubits: Optional[int] = None, name: str = 'tridi') -> None:
        super().__init__(name=name)

        self._num_state_qubits = None

        self._tolerance = tolerance if tolerance is not None else 1e-2
        self._time = time if time is not None else 1

        self.num_state_qubits = num_state_qubits

    def _build(self):

    def power(self):
    def control(self):
