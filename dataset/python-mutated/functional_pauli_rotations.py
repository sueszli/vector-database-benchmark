"""Base class for functional Pauli rotations."""
from typing import Optional
from abc import ABC, abstractmethod
from ..blueprintcircuit import BlueprintCircuit

class FunctionalPauliRotations(BlueprintCircuit, ABC):
    """Base class for functional Pauli rotations."""

    def __init__(self, num_state_qubits: Optional[int]=None, basis: str='Y', name: str='F') -> None:
        if False:
            return 10
        "Create a new functional Pauli rotation circuit.\n\n        Args:\n            num_state_qubits: The number of qubits representing the state :math:`|x\\rangle`.\n            basis: The kind of Pauli rotation to use. Must be 'X', 'Y' or 'Z'.\n            name: The name of the circuit object.\n        "
        super().__init__(name=name)
        self._num_state_qubits = None
        self._basis = None
        self.num_state_qubits = num_state_qubits
        self.basis = basis

    @property
    def basis(self) -> str:
        if False:
            print('Hello World!')
        "The kind of Pauli rotation to be used.\n\n        Set the basis to 'X', 'Y' or 'Z' for controlled-X, -Y, or -Z rotations respectively.\n\n        Returns:\n            The kind of Pauli rotation used in controlled rotation.\n        "
        return self._basis

    @basis.setter
    def basis(self, basis: str) -> None:
        if False:
            i = 10
            return i + 15
        'Set the kind of Pauli rotation to be used.\n\n        Args:\n            basis: The Pauli rotation to be used.\n\n        Raises:\n            ValueError: The provided basis in not X, Y or Z.\n        '
        basis = basis.lower()
        if self._basis is None or basis != self._basis:
            if basis not in ['x', 'y', 'z']:
                raise ValueError(f'The provided basis must be X, Y or Z, not {basis}')
            self._invalidate()
            self._basis = basis

    @property
    def num_state_qubits(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The number of state qubits representing the state :math:`|x\\rangle`.\n\n        Returns:\n            The number of state qubits.\n        '
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: Optional[int]) -> None:
        if False:
            i = 10
            return i + 15
        'Set the number of state qubits.\n\n        Note that this may change the underlying quantum register, if the number of state qubits\n        changes.\n\n        Args:\n            num_state_qubits: The new number of qubits.\n        '
        if self._num_state_qubits is None or num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits
            self._reset_registers(num_state_qubits)

    @abstractmethod
    def _reset_registers(self, num_state_qubits: Optional[int]) -> None:
        if False:
            print('Hello World!')
        'Reset the registers according to the new number of state qubits.\n\n        Args:\n            num_state_qubits: The new number of qubits.\n        '
        raise NotImplementedError

    @property
    def num_ancilla_qubits(self) -> int:
        if False:
            while True:
                i = 10
        'The minimum number of ancilla qubits in the circuit.\n\n        Returns:\n            The minimal number of ancillas required.\n        '
        return 0