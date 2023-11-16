"""
This is the Parametric Circuit class: anything that you need for a circuit
to be parametrized and used for approximate compiling optimization.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from .approximate import ApproximateCircuit

class CNOTUnitCircuit(ApproximateCircuit):
    """A class that represents an approximate circuit based on CNOT unit blocks."""

    def __init__(self, num_qubits: int, cnots: np.ndarray, tol: Optional[float]=0.0, name: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            num_qubits: the number of qubits in this circuit.\n            cnots: an array of dimensions ``(2, L)`` indicating where the CNOT units will be placed.\n            tol: angle parameter less or equal this (small) value is considered equal zero and\n                corresponding gate is not inserted into the output circuit (because it becomes\n                identity one in this case).\n            name: name of this circuit\n\n        Raises:\n            ValueError: if an unsupported parameter is passed.\n        '
        super().__init__(num_qubits=num_qubits, name=name)
        if cnots.ndim != 2 or cnots.shape[0] != 2:
            raise ValueError('CNOT structure must be defined as an array of the size (2, N)')
        self._cnots = cnots
        self._num_cnots = cnots.shape[1]
        self._tol = tol
        self._thetas: np.ndarray | None = None

    @property
    def thetas(self) -> np.ndarray:
        if False:
            return 10
        '\n        Returns a vector of rotation angles used by CNOT units in this circuit.\n\n        Returns:\n            Parameters of the rotation gates in this circuit.\n        '
        return self._thetas

    def build(self, thetas: np.ndarray) -> None:
        if False:
            while True:
                i = 10
        '\n        Constructs a Qiskit quantum circuit out of the parameters (angles) of this circuit. If a\n            parameter value is less in absolute value than the specified tolerance then the\n            corresponding rotation gate will be skipped in the circuit.\n        '
        n = self.num_qubits
        self._thetas = thetas
        cnots = self._cnots
        for k in range(n):
            p = 4 * self._num_cnots + 3 * k
            k = n - k - 1
            if np.abs(thetas[2 + p]) > self._tol:
                self.rz(thetas[2 + p], k)
            if np.abs(thetas[1 + p]) > self._tol:
                self.ry(thetas[1 + p], k)
            if np.abs(thetas[0 + p]) > self._tol:
                self.rz(thetas[0 + p], k)
        for c in range(self._num_cnots):
            p = 4 * c
            q1 = n - 1 - int(cnots[0, c])
            q2 = n - 1 - int(cnots[1, c])
            self.cx(q1, q2)
            if np.abs(thetas[0 + p]) > self._tol:
                self.ry(thetas[0 + p], q1)
            if np.abs(thetas[1 + p]) > self._tol:
                self.rz(thetas[1 + p], q1)
            if np.abs(thetas[2 + p]) > self._tol:
                self.ry(thetas[2 + p], q2)
            if np.abs(thetas[3 + p]) > self._tol:
                self.rx(thetas[3 + p], q2)