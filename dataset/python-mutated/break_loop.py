"""Circuit operation representing a ``break`` from a loop."""
from typing import Optional
from qiskit.circuit.instruction import Instruction
from .builder import InstructionPlaceholder, InstructionResources

class BreakLoopOp(Instruction):
    """A circuit operation which, when encountered, jumps to the end of
    the nearest enclosing loop.

    .. note:

        Can be inserted only within the body of a loop op, and must span
        the full width of that block.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────────┐
        q_0: ┤0             ├
             │              │
        q_1: ┤1             ├
             │  break_loop  │
        q_2: ┤2             ├
             │              │
        c_0: ╡0             ╞
             └──────────────┘

    """

    def __init__(self, num_qubits: int, num_clbits: int, label: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('break_loop', num_qubits, num_clbits, [], label=label)

class BreakLoopPlaceholder(InstructionPlaceholder):
    """A placeholder instruction for use in control-flow context managers, when the number of qubits
    and clbits is not yet known.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(self, *, label: Optional[str]=None):
        if False:
            return 10
        super().__init__('break_loop', 0, 0, [], label=label)

    def concrete_instruction(self, qubits, clbits):
        if False:
            return 10
        return (self._copy_mutable_properties(BreakLoopOp(len(qubits), len(clbits), label=self.label)), InstructionResources(qubits=tuple(qubits), clbits=tuple(clbits)))

    def placeholder_resources(self):
        if False:
            for i in range(10):
                print('nop')
        return InstructionResources()