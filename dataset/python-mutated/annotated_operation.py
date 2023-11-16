"""Annotated Operations."""
import dataclasses
from typing import Union, List
from qiskit.circuit.operation import Operation
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.exceptions import CircuitError

class Modifier:
    """The base class that all modifiers of :class:`~.AnnotatedOperation` should
    inherit from."""
    pass

@dataclasses.dataclass
class InverseModifier(Modifier):
    """Inverse modifier: specifies that the operation is inverted."""
    pass

@dataclasses.dataclass
class ControlModifier(Modifier):
    """Control modifier: specifies that the operation is controlled by ``num_ctrl_qubits``
    and has control state ``ctrl_state``."""
    num_ctrl_qubits: int = 0
    ctrl_state: Union[int, str, None] = None

    def __init__(self, num_ctrl_qubits: int=0, ctrl_state: Union[int, str, None]=None):
        if False:
            for i in range(10):
                print('nop')
        self.num_ctrl_qubits = num_ctrl_qubits
        self.ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)

@dataclasses.dataclass
class PowerModifier(Modifier):
    """Power modifier: specifies that the operation is raised to the power ``power``."""
    power: float

class AnnotatedOperation(Operation):
    """Annotated operation."""

    def __init__(self, base_op: Operation, modifiers: Union[Modifier, List[Modifier]]):
        if False:
            while True:
                i = 10
        '\n        Create a new AnnotatedOperation.\n\n        An "annotated operation" allows to add a list of modifiers to the\n        "base" operation. For now, the only supported modifiers are of\n        types :class:`~.InverseModifier`, :class:`~.ControlModifier` and\n        :class:`~.PowerModifier`.\n\n        An annotated operation can be viewed as an extension of\n        :class:`~.ControlledGate` (which also allows adding control to the\n        base operation). However, an important difference is that the\n        circuit definition of an annotated operation is not constructed when\n        the operation is declared, and instead happens during transpilation,\n        specifically during the :class:`~.HighLevelSynthesis` transpiler pass.\n\n        An annotated operation can be also viewed as a "higher-level"\n        or "more abstract" object that can be added to a quantum circuit.\n        This enables writing transpiler optimization passes that make use of\n        this higher-level representation, for instance removing a gate\n        that is immediately followed by its inverse.\n\n        Args:\n            base_op: base operation being modified\n            modifiers: ordered list of modifiers. Supported modifiers include\n                ``InverseModifier``, ``ControlModifier`` and ``PowerModifier``.\n\n        Examples::\n\n            op1 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(2)])\n\n            op2_inner = AnnotatedGate(SGate(), InverseModifier())\n            op2 = AnnotatedGate(op2_inner, ControlModifier(2))\n\n        Both op1 and op2 are semantically equivalent to an ``SGate()`` which is first\n        inverted and then controlled by 2 qubits.\n        '
        self.base_op = base_op
        self.modifiers = modifiers if isinstance(modifiers, List) else [modifiers]

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        'Unique string identifier for operation type.'
        return 'annotated'

    @property
    def num_qubits(self):
        if False:
            i = 10
            return i + 15
        'Number of qubits.'
        num_ctrl_qubits = 0
        for modifier in self.modifiers:
            if isinstance(modifier, ControlModifier):
                num_ctrl_qubits += modifier.num_ctrl_qubits
        return num_ctrl_qubits + self.base_op.num_qubits

    @property
    def num_clbits(self):
        if False:
            return 10
        'Number of classical bits.'
        return self.base_op.num_clbits

    def __eq__(self, other) -> bool:
        if False:
            return 10
        'Checks if two AnnotatedOperations are equal.'
        return isinstance(other, AnnotatedOperation) and self.modifiers == other.modifiers and (self.base_op == other.base_op)

    def copy(self) -> 'AnnotatedOperation':
        if False:
            while True:
                i = 10
        'Return a copy of the :class:`~.AnnotatedOperation`.'
        return AnnotatedOperation(base_op=self.base_op, modifiers=self.modifiers.copy())

    def to_matrix(self):
        if False:
            while True:
                i = 10
        'Return a matrix representation (allowing to construct Operator).'
        from qiskit.quantum_info.operators import Operator
        operator = Operator(self.base_op)
        for modifier in self.modifiers:
            if isinstance(modifier, InverseModifier):
                operator = operator.power(-1)
            elif isinstance(modifier, ControlModifier):
                operator = Operator(_compute_control_matrix(operator.data, modifier.num_ctrl_qubits, modifier.ctrl_state))
            elif isinstance(modifier, PowerModifier):
                operator = operator.power(modifier.power)
            else:
                raise CircuitError(f'Unknown modifier {modifier}.')
        return operator

def _canonicalize_modifiers(modifiers):
    if False:
        while True:
            i = 10
    '\n    Returns the canonical representative of the modifier list. This is possible\n    since all the modifiers commute; also note that InverseModifier is a special\n    case of PowerModifier. The current solution is to compute the total number\n    of control qubits / control state and the total power. The InverseModifier\n    will be present if total power is negative, whereas the power modifier will\n    be present only with positive powers different from 1.\n    '
    power = 1
    num_ctrl_qubits = 0
    ctrl_state = 0
    for modifier in modifiers:
        if isinstance(modifier, InverseModifier):
            power *= -1
        elif isinstance(modifier, ControlModifier):
            num_ctrl_qubits += modifier.num_ctrl_qubits
            ctrl_state = ctrl_state << modifier.num_ctrl_qubits | modifier.ctrl_state
        elif isinstance(modifier, PowerModifier):
            power *= modifier.power
        else:
            raise CircuitError(f'Unknown modifier {modifier}.')
    canonical_modifiers = []
    if power < 0:
        canonical_modifiers.append(InverseModifier())
        power *= -1
    if power != 1:
        canonical_modifiers.append(PowerModifier(power))
    if num_ctrl_qubits > 0:
        canonical_modifiers.append(ControlModifier(num_ctrl_qubits, ctrl_state))
    return canonical_modifiers