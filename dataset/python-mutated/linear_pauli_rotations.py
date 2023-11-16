"""Linearly-controlled X, Y or Z rotation."""
from typing import Optional
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations

class LinearPauliRotations(FunctionalPauliRotations):
    """Linearly-controlled X, Y or Z rotation.

    For a register of state qubits :math:`|x\\rangle`, a target qubit :math:`|0\\rangle` and the
    basis ``'Y'`` this circuit acts as:

    .. parsed-literal::

            q_0: ─────────────────────────■───────── ... ──────────────────────
                                          │
                                          .
                                          │
        q_(n-1): ─────────────────────────┼───────── ... ───────────■──────────
                  ┌────────────┐  ┌───────┴───────┐       ┌─────────┴─────────┐
            q_n: ─┤ RY(offset) ├──┤ RY(2^0 slope) ├  ...  ┤ RY(2^(n-1) slope) ├
                  └────────────┘  └───────────────┘       └───────────────────┘

    This can for example be used to approximate linear functions, with :math:`a =` ``slope``:math:`/2`
    and :math:`b =` ``offset``:math:`/2` and the basis ``'Y'``:

    .. math::

        |x\\rangle |0\\rangle \\mapsto \\cos(ax + b)|x\\rangle|0\\rangle + \\sin(ax + b)|x\\rangle |1\\rangle

    Since for small arguments :math:`\\sin(x) \\approx x` this operator can be used to approximate
    linear functions.
    """

    def __init__(self, num_state_qubits: Optional[int]=None, slope: float=1, offset: float=0, basis: str='Y', name: str='LinRot') -> None:
        if False:
            return 10
        "Create a new linear rotation circuit.\n\n        Args:\n            num_state_qubits: The number of qubits representing the state :math:`|x\\rangle`.\n            slope: The slope of the controlled rotation.\n            offset: The offset of the controlled rotation.\n            basis: The type of Pauli rotation ('X', 'Y', 'Z').\n            name: The name of the circuit object.\n        "
        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)
        self._slope = None
        self._offset = None
        self.slope = slope
        self.offset = offset

    @property
    def slope(self) -> float:
        if False:
            while True:
                i = 10
        'The multiplicative factor in the rotation angle of the controlled rotations.\n\n        The rotation angles are ``slope * 2^0``, ``slope * 2^1``, ... , ``slope * 2^(n-1)`` where\n        ``n`` is the number of state qubits.\n\n        Returns:\n            The rotation angle common in all controlled rotations.\n        '
        return self._slope

    @slope.setter
    def slope(self, slope: float) -> None:
        if False:
            while True:
                i = 10
        'Set the multiplicative factor of the rotation angles.\n\n        Args:\n            The slope of the rotation angles.\n        '
        if self._slope is None or slope != self._slope:
            self._invalidate()
            self._slope = slope

    @property
    def offset(self) -> float:
        if False:
            while True:
                i = 10
        'The angle of the single qubit offset rotation on the target qubit.\n\n        Before applying the controlled rotations, a single rotation of angle ``offset`` is\n        applied to the target qubit.\n\n        Returns:\n            The offset angle.\n        '
        return self._offset

    @offset.setter
    def offset(self, offset: float) -> None:
        if False:
            print('Hello World!')
        'Set the angle for the offset rotation on the target qubit.\n\n        Args:\n            offset: The offset rotation angle.\n        '
        if self._offset is None or offset != self._offset:
            self._invalidate()
            self._offset = offset

    def _reset_registers(self, num_state_qubits: Optional[int]) -> None:
        if False:
            print('Hello World!')
        'Set the number of state qubits.\n\n        Note that this changes the underlying quantum register, if the number of state qubits\n        changes.\n\n        Args:\n            num_state_qubits: The new number of qubits.\n        '
        self.qregs = []
        if num_state_qubits:
            qr_state = QuantumRegister(num_state_qubits, name='state')
            qr_target = QuantumRegister(1, name='target')
            self.qregs = [qr_state, qr_target]

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
        return valid

    def _build(self):
        if False:
            return 10
        'If not already built, build the circuit.'
        if self._is_built:
            return
        super()._build()
        circuit = QuantumCircuit(*self.qregs, name=self.name)
        qr_state = self.qubits[:self.num_state_qubits]
        qr_target = self.qubits[self.num_state_qubits]
        if self.basis == 'x':
            circuit.rx(self.offset, qr_target)
        elif self.basis == 'y':
            circuit.ry(self.offset, qr_target)
        else:
            circuit.rz(self.offset, qr_target)
        for (i, q_i) in enumerate(qr_state):
            if self.basis == 'x':
                circuit.crx(self.slope * pow(2, i), q_i, qr_target)
            elif self.basis == 'y':
                circuit.cry(self.slope * pow(2, i), q_i, qr_target)
            else:
                circuit.crz(self.slope * pow(2, i), q_i, qr_target)
        self.append(circuit.to_gate(), self.qubits)