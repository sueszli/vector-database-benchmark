"""Evolution synthesis."""
from abc import ABC, abstractmethod
from typing import Any, Dict

class EvolutionSynthesis(ABC):
    """Interface for evolution synthesis algorithms."""

    @abstractmethod
    def synthesize(self, evolution):
        if False:
            return 10
        'Synthesize an ``qiskit.circuit.library.PauliEvolutionGate``.\n\n        Args:\n            evolution (PauliEvolutionGate): The evolution gate to synthesize.\n\n        Returns:\n            QuantumCircuit: A circuit implementing the evolution.\n        '
        raise NotImplementedError

    @property
    def settings(self) -> Dict[str, Any]:
        if False:
            return 10
        'Return the settings in a dictionary, which can be used to reconstruct the object.\n\n        Returns:\n            A dictionary containing the settings of this product formula.\n\n        Raises:\n            NotImplementedError: The interface does not implement this method.\n        '
        raise NotImplementedError('The settings property is not implemented for the base interface.')