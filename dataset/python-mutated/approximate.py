"""Base classes for an approximate circuit definition."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, SupportsFloat
import numpy as np
from qiskit import QuantumCircuit

class ApproximateCircuit(QuantumCircuit, ABC):
    """A base class that represents an approximate circuit."""

    def __init__(self, num_qubits: int, name: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            num_qubits: number of qubit this circuit will span.\n            name: a name of the circuit.\n        '
        super().__init__(num_qubits, name=name)

    @property
    @abstractmethod
    def thetas(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        The property is not implemented and raises a ``NotImplementedException`` exception.\n\n        Returns:\n            a vector of parameters of this circuit.\n        '
        raise NotImplementedError

    @abstractmethod
    def build(self, thetas: np.ndarray) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Constructs this circuit out of the parameters(thetas). Parameter values must be set before\n            constructing the circuit.\n\n        Args:\n            thetas: a vector of parameters to be set in this circuit.\n        '
        raise NotImplementedError

class ApproximatingObjective(ABC):
    """
    A base class for an optimization problem definition. An implementing class must provide at least
    an implementation of the ``objective`` method. In such case only gradient free optimizers can
    be used. Both method, ``objective`` and ``gradient``, preferable to have in an implementation.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._target_matrix: np.ndarray | None = None

    @abstractmethod
    def objective(self, param_values: np.ndarray) -> SupportsFloat:
        if False:
            return 10
        '\n        Computes a value of the objective function given a vector of parameter values.\n\n        Args:\n            param_values: a vector of parameter values for the optimization problem.\n\n        Returns:\n            a float value of the objective function.\n        '
        raise NotImplementedError

    @abstractmethod
    def gradient(self, param_values: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Computes a gradient with respect to parameters given a vector of parameter values.\n\n        Args:\n            param_values: a vector of parameter values for the optimization problem.\n\n        Returns:\n            an array of gradient values.\n        '
        raise NotImplementedError

    @property
    def target_matrix(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            a matrix being approximated\n        '
        return self._target_matrix

    @target_matrix.setter
    def target_matrix(self, target_matrix: np.ndarray) -> None:
        if False:
            return 10
        '\n        Args:\n            target_matrix: a matrix to approximate in the optimization procedure.\n        '
        self._target_matrix = target_matrix

    @property
    @abstractmethod
    def num_thetas(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n\n        Returns:\n            the number of parameters in this optimization problem.\n        '
        raise NotImplementedError