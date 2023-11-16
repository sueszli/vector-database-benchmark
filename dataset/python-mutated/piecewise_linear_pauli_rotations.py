"""Piecewise-linearly-controlled rotation."""
from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
from .linear_pauli_rotations import LinearPauliRotations
from .integer_comparator import IntegerComparator

class PiecewiseLinearPauliRotations(FunctionalPauliRotations):
    """Piecewise-linearly-controlled Pauli rotations.

    For a piecewise linear (not necessarily continuous) function :math:`f(x)`, which is defined
    through breakpoints, slopes and offsets as follows.
    Suppose the breakpoints :math:`(x_0, ..., x_J)` are a subset of :math:`[0, 2^n-1]`, where
    :math:`n` is the number of state qubits. Further on, denote the corresponding slopes and
    offsets by :math:`a_j` and :math:`b_j` respectively.
    Then f(x) is defined as:

    .. math::

        f(x) = \\begin{cases}
            0, x < x_0 \\\\
            a_j (x - x_j) + b_j, x_j \\leq x < x_{j+1}
            \\end{cases}

    where we implicitly assume :math:`x_{J+1} = 2^n`.
    """

    def __init__(self, num_state_qubits: int | None=None, breakpoints: list[int] | None=None, slopes: list[float] | np.ndarray | None=None, offsets: list[float] | np.ndarray | None=None, basis: str='Y', name: str='pw_lin') -> None:
        if False:
            return 10
        "Construct piecewise-linearly-controlled Pauli rotations.\n\n        Args:\n            num_state_qubits: The number of qubits representing the state.\n            breakpoints: The breakpoints to define the piecewise-linear function.\n                Defaults to ``[0]``.\n            slopes: The slopes for different segments of the piecewise-linear function.\n                Defaults to ``[1]``.\n            offsets: The offsets for different segments of the piecewise-linear function.\n                Defaults to ``[0]``.\n            basis: The type of Pauli rotation (``'X'``, ``'Y'``, ``'Z'``).\n            name: The name of the circuit.\n        "
        self._breakpoints = breakpoints if breakpoints is not None else [0]
        self._slopes = slopes if slopes is not None else [1]
        self._offsets = offsets if offsets is not None else [0]
        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)

    @property
    def breakpoints(self) -> list[int]:
        if False:
            print('Hello World!')
        'The breakpoints of the piecewise linear function.\n\n        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last\n        point implicitly is ``2**(num_state_qubits + 1)``.\n        '
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints: list[int]) -> None:
        if False:
            while True:
                i = 10
        'Set the breakpoints.\n\n        Args:\n            breakpoints: The new breakpoints.\n        '
        self._invalidate()
        self._breakpoints = breakpoints
        if self.num_state_qubits and breakpoints:
            self._reset_registers(self.num_state_qubits)

    @property
    def slopes(self) -> list[float] | np.ndarray:
        if False:
            i = 10
            return i + 15
        'The breakpoints of the piecewise linear function.\n\n        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last\n        point implicitly is ``2**(num_state_qubits + 1)``.\n        '
        return self._slopes

    @slopes.setter
    def slopes(self, slopes: list[float]) -> None:
        if False:
            print('Hello World!')
        'Set the slopes.\n\n        Args:\n            slopes: The new slopes.\n        '
        self._invalidate()
        self._slopes = slopes

    @property
    def offsets(self) -> list[float] | np.ndarray:
        if False:
            print('Hello World!')
        'The breakpoints of the piecewise linear function.\n\n        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last\n        point implicitly is ``2**(num_state_qubits + 1)``.\n        '
        return self._offsets

    @offsets.setter
    def offsets(self, offsets: list[float]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the offsets.\n\n        Args:\n            offsets: The new offsets.\n        '
        self._invalidate()
        self._offsets = offsets

    @property
    def mapped_slopes(self) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'The slopes mapped to the internal representation.\n\n        Returns:\n            The mapped slopes.\n        '
        mapped_slopes = np.zeros_like(self.slopes)
        for (i, slope) in enumerate(self.slopes):
            mapped_slopes[i] = slope - sum(mapped_slopes[:i])
        return mapped_slopes

    @property
    def mapped_offsets(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        'The offsets mapped to the internal representation.\n\n        Returns:\n            The mapped offsets.\n        '
        mapped_offsets = np.zeros_like(self.offsets)
        for (i, (offset, slope, point)) in enumerate(zip(self.offsets, self.slopes, self.breakpoints)):
            mapped_offsets[i] = offset - slope * point - sum(mapped_offsets[:i])
        return mapped_offsets

    @property
    def contains_zero_breakpoint(self) -> bool | np.bool_:
        if False:
            print('Hello World!')
        'Whether 0 is the first breakpoint.\n\n        Returns:\n            True, if 0 is the first breakpoint, otherwise False.\n        '
        return np.isclose(0, self.breakpoints[0])

    def evaluate(self, x: float) -> float:
        if False:
            return 10
        'Classically evaluate the piecewise linear rotation.\n\n        Args:\n            x: Value to be evaluated at.\n\n        Returns:\n            Value of piecewise linear function at x.\n        '
        y = (x >= self.breakpoints[0]) * (x * self.mapped_slopes[0] + self.mapped_offsets[0])
        for i in range(1, len(self.breakpoints)):
            y = y + (x >= self.breakpoints[i]) * (x * self.mapped_slopes[i] + self.mapped_offsets[i])
        return y

    def _check_configuration(self, raise_on_failure: bool=True) -> bool:
        if False:
            return 10
        'Check if the current configuration is valid.'
        valid = True
        if self.num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of qubits has not been set.')
        if self.num_qubits < self.num_state_qubits + 1:
            valid = False
            if raise_on_failure:
                raise CircuitError('Not enough qubits in the circuit, need at least {}.'.format(self.num_state_qubits + 1))
        if len(self.breakpoints) != len(self.slopes) or len(self.breakpoints) != len(self.offsets):
            valid = False
            if raise_on_failure:
                raise ValueError('Mismatching sizes of breakpoints, slopes and offsets.')
        return valid

    def _reset_registers(self, num_state_qubits: int | None) -> None:
        if False:
            while True:
                i = 10
        'Reset the registers.'
        self.qregs = []
        if num_state_qubits is not None:
            qr_state = QuantumRegister(num_state_qubits)
            qr_target = QuantumRegister(1)
            self.qregs = [qr_state, qr_target]
            if len(self.breakpoints) > 1:
                num_ancillas = num_state_qubits
                qr_ancilla = AncillaRegister(num_ancillas)
                self.add_register(qr_ancilla)

    def _build(self):
        if False:
            while True:
                i = 10
        'If not already built, build the circuit.'
        if self._is_built:
            return
        super()._build()
        circuit = QuantumCircuit(*self.qregs, name=self.name)
        qr_state = circuit.qubits[:self.num_state_qubits]
        qr_target = [circuit.qubits[self.num_state_qubits]]
        qr_ancilla = circuit.ancillas
        for (i, point) in enumerate(self.breakpoints):
            if i == 0 and self.contains_zero_breakpoint:
                lin_r = LinearPauliRotations(num_state_qubits=self.num_state_qubits, slope=self.mapped_slopes[i], offset=self.mapped_offsets[i], basis=self.basis)
                circuit.append(lin_r.to_gate(), qr_state[:] + qr_target)
            else:
                qr_compare = [qr_ancilla[0]]
                qr_helper = qr_ancilla[1:]
                comp = IntegerComparator(num_state_qubits=self.num_state_qubits, value=point)
                qr = qr_state[:] + qr_compare[:]
                circuit.append(comp.to_gate(), qr[:] + qr_helper[:comp.num_ancillas])
                lin_r = LinearPauliRotations(num_state_qubits=self.num_state_qubits, slope=self.mapped_slopes[i], offset=self.mapped_offsets[i], basis=self.basis)
                circuit.append(lin_r.to_gate().control(), qr_compare[:] + qr_state[:] + qr_target)
                circuit.append(comp.to_gate().inverse(), qr[:] + qr_helper[:comp.num_ancillas])
        self.append(circuit.to_gate(), self.qubits)