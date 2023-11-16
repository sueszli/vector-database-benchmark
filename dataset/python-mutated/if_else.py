"""Circuit operation representing an ``if/else`` statement."""
from __future__ import annotations
from typing import Optional, Union, Iterable
import itertools
from qiskit.circuit import ClassicalRegister, Clbit, QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.exceptions import CircuitError
from .builder import ControlFlowBuilderBlock, InstructionPlaceholder, InstructionResources
from .control_flow import ControlFlowOp
from ._builder_utils import partition_registers, unify_circuit_resources, validate_condition, condition_resources
__all__ = ('IfElseOp',)

class IfElseOp(ControlFlowOp):
    """A circuit operation which executes a program (``true_body``) if a
    provided condition (``condition``) evaluates to true, and
    optionally evaluates another program (``false_body``) otherwise.

    Parameters:
        condition: A condition to be evaluated at circuit runtime which,
            if true, will trigger the evaluation of ``true_body``. Can be
            specified as either a tuple of a ``ClassicalRegister`` to be
            tested for equality with a given ``int``, or as a tuple of a
            ``Clbit`` to be compared to either a ``bool`` or an ``int``.
        true_body: A program to be executed if ``condition`` evaluates
            to true.
        false_body: A optional program to be executed if ``condition``
            evaluates to false.
        label: An optional label for identifying the instruction.

    If provided, ``false_body`` must be of the same ``num_qubits`` and
    ``num_clbits`` as ``true_body``.

    The classical bits used in ``condition`` must be a subset of those attached
    to the circuit on which this ``IfElseOp`` will be appended.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1          ├
             │  if_else  │
        q_2: ┤2          ├
             │           │
        c_0: ╡0          ╞
             └───────────┘

    """

    def __init__(self, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, true_body: QuantumCircuit, false_body: QuantumCircuit | None=None, label: str | None=None):
        if False:
            while True:
                i = 10
        if not isinstance(true_body, QuantumCircuit):
            raise CircuitError(f'IfElseOp expects a true_body parameter of type QuantumCircuit, but received {type(true_body)}.')
        num_qubits = true_body.num_qubits
        num_clbits = true_body.num_clbits
        super().__init__('if_else', num_qubits, num_clbits, [true_body, false_body], label=label)
        self.condition = validate_condition(condition)

    @property
    def params(self):
        if False:
            return 10
        return self._params

    @params.setter
    def params(self, parameters):
        if False:
            print('Hello World!')
        (true_body, false_body) = parameters
        if not isinstance(true_body, QuantumCircuit):
            raise CircuitError(f'IfElseOp expects a true_body parameter of type QuantumCircuit, but received {type(true_body)}.')
        if true_body.num_qubits != self.num_qubits or true_body.num_clbits != self.num_clbits:
            raise CircuitError(f'Attempted to assign a true_body parameter with a num_qubits or num_clbits different than that of the IfElseOp. IfElseOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} Supplied body num_qubits/clbits: {true_body.num_qubits}/{true_body.num_clbits}.')
        if false_body is not None:
            if not isinstance(false_body, QuantumCircuit):
                raise CircuitError(f'IfElseOp expects a false_body parameter of type QuantumCircuit, but received {type(false_body)}.')
            if false_body.num_qubits != self.num_qubits or false_body.num_clbits != self.num_clbits:
                raise CircuitError(f'Attempted to assign a false_body parameter with a num_qubits or num_clbits different than that of the IfElseOp. IfElseOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} Supplied body num_qubits/clbits: {false_body.num_qubits}/{false_body.num_clbits}.')
        self._params = [true_body, false_body]

    @property
    def blocks(self):
        if False:
            i = 10
            return i + 15
        if self.params[1] is None:
            return (self.params[0],)
        else:
            return (self.params[0], self.params[1])

    def replace_blocks(self, blocks: Iterable[QuantumCircuit]) -> 'IfElseOp':
        if False:
            for i in range(10):
                print('nop')
        'Replace blocks and return new instruction.\n\n        Args:\n            blocks: Iterable of circuits for "if" and "else" condition. If there is no "else"\n                circuit it may be set to None or omitted.\n\n        Returns:\n            New IfElseOp with replaced blocks.\n        '
        (true_body, false_body) = (ablock for (ablock, _) in itertools.zip_longest(blocks, range(2), fillvalue=None))
        return IfElseOp(self.condition, true_body, false_body=false_body, label=self.label)

    def c_if(self, classical, val):
        if False:
            print('Hello World!')
        raise NotImplementedError('IfElseOp cannot be classically controlled through Instruction.c_if. Please nest it in an IfElseOp instead.')

class IfElsePlaceholder(InstructionPlaceholder):
    """A placeholder instruction to use in control-flow context managers, when calculating the
    number of resources this instruction should block is deferred until the construction of the
    outer loop.

    This generally should not be instantiated manually; only :obj:`.IfContext` and
    :obj:`.ElseContext` should do it when they need to defer creation of the concrete instruction.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(self, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, true_block: ControlFlowBuilderBlock, false_block: ControlFlowBuilderBlock | None=None, *, label: Optional[str]=None):
        if False:
            print('Hello World!')
        '\n        Args:\n            condition: the condition to execute the true block on.  This has the same semantics as\n                the ``condition`` argument to :obj:`.IfElseOp`.\n            true_block: the unbuilt scope block that will become the "true" branch at creation time.\n            false_block: if given, the unbuilt scope block that will become the "false" branch at\n                creation time.\n            label: the label to give the operator when it is created.\n        '
        self.__true_block = true_block
        self.__false_block: Optional[ControlFlowBuilderBlock] = false_block
        self.__resources = self._calculate_placeholder_resources()
        super().__init__('if_else', len(self.__resources.qubits), len(self.__resources.clbits), [], label=label)
        self.condition = validate_condition(condition)

    def with_false_block(self, false_block: ControlFlowBuilderBlock) -> 'IfElsePlaceholder':
        if False:
            while True:
                i = 10
        'Return a new placeholder instruction, with the false block set to the given value,\n        updating the bits used by both it and the true body, if necessary.\n\n        It is an error to try and set the false block on a placeholder that already has one.\n\n        Args:\n            false_block: The (unbuilt) instruction scope to set the false body to.\n\n        Returns:\n            A new placeholder, with ``false_block`` set to the given input, and both true and false\n            blocks expanded to account for all resources.\n\n        Raises:\n            CircuitError: if the false block of this placeholder instruction is already set.\n        '
        if self.__false_block is not None:
            raise CircuitError(f'false block is already set to {self.__false_block}')
        true_block = self.__true_block.copy()
        true_bits = true_block.qubits | true_block.clbits
        false_bits = false_block.qubits | false_block.clbits
        true_block.add_bits(false_bits - true_bits)
        false_block.add_bits(true_bits - false_bits)
        return type(self)(self.condition, true_block, false_block, label=self.label)

    def registers(self):
        if False:
            return 10
        'Get the registers used by the interior blocks.'
        if self.__false_block is None:
            return self.__true_block.registers.copy()
        return self.__true_block.registers | self.__false_block.registers

    def _calculate_placeholder_resources(self) -> InstructionResources:
        if False:
            return 10
        'Get the placeholder resources (see :meth:`.placeholder_resources`).\n\n        This is a separate function because we use the resources during the initialisation to\n        determine how we should set our ``num_qubits`` and ``num_clbits``, so we implement the\n        public version as a cache access for efficiency.\n        '
        if self.__false_block is None:
            (qregs, cregs) = partition_registers(self.__true_block.registers)
            return InstructionResources(qubits=tuple(self.__true_block.qubits), clbits=tuple(self.__true_block.clbits), qregs=tuple(qregs), cregs=tuple(cregs))
        (true_qregs, true_cregs) = partition_registers(self.__true_block.registers)
        (false_qregs, false_cregs) = partition_registers(self.__false_block.registers)
        return InstructionResources(qubits=tuple(self.__true_block.qubits | self.__false_block.qubits), clbits=tuple(self.__true_block.clbits | self.__false_block.clbits), qregs=tuple(true_qregs) + tuple(false_qregs), cregs=tuple(true_cregs) + tuple(false_cregs))

    def placeholder_resources(self):
        if False:
            while True:
                i = 10
        return self.__resources

    def concrete_instruction(self, qubits, clbits):
        if False:
            while True:
                i = 10
        current_qubits = self.__true_block.qubits
        current_clbits = self.__true_block.clbits
        if self.__false_block is not None:
            current_qubits = current_qubits | self.__false_block.qubits
            current_clbits = current_clbits | self.__false_block.clbits
        all_bits = qubits | clbits
        current_bits = current_qubits | current_clbits
        if current_bits - all_bits:
            raise CircuitError(f'This block contains bits that are not in the operands sets: {current_bits - all_bits!r}')
        true_body = self.__true_block.build(qubits, clbits)
        if self.__false_block is None:
            false_body = None
        else:
            (true_body, false_body) = unify_circuit_resources((true_body, self.__false_block.build(qubits, clbits)))
        return (self._copy_mutable_properties(IfElseOp(self.condition, true_body, false_body, label=self.label)), InstructionResources(qubits=tuple(true_body.qubits), clbits=tuple(true_body.clbits), qregs=tuple(true_body.qregs), cregs=tuple(true_body.cregs)))

    def c_if(self, classical, val):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('IfElseOp cannot be classically controlled through Instruction.c_if. Please nest it in another IfElseOp instead.')

class IfContext:
    """A context manager for building up ``if`` statements onto circuits in a natural order, without
    having to construct the statement body first.

    The return value of this context manager can be used immediately following the block to create
    an attached ``else`` statement.

    This context should almost invariably be created by a :meth:`.QuantumCircuit.if_test` call, and
    the resulting instance is a "friend" of the calling circuit.  The context will manipulate the
    circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and exited
    (by popping its scope, building it, and appending the resulting :obj:`.IfElseOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    __slots__ = ('_appended_instructions', '_circuit', '_condition', '_in_loop', '_label')

    def __init__(self, circuit: QuantumCircuit, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, *, in_loop: bool, label: str | None=None):
        if False:
            print('Hello World!')
        self._circuit = circuit
        self._condition = validate_condition(condition)
        self._label = label
        self._appended_instructions = None
        self._in_loop = in_loop

    @property
    def circuit(self) -> QuantumCircuit:
        if False:
            print('Hello World!')
        'Get the circuit that this context manager is attached to.'
        return self._circuit

    @property
    def condition(self) -> tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr:
        if False:
            while True:
                i = 10
        'Get the expression that this statement is conditioned on.'
        return self._condition

    @property
    def appended_instructions(self) -> Union[InstructionSet, None]:
        if False:
            i = 10
            return i + 15
        'Get the instruction set that was created when this block finished.  If the block has not\n        yet finished, then this will be ``None``.'
        return self._appended_instructions

    @property
    def in_loop(self) -> bool:
        if False:
            return 10
        'Whether this context manager is enclosed within a loop.'
        return self._in_loop

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        resources = condition_resources(self._condition)
        self._circuit._push_scope(clbits=resources.clbits, registers=resources.cregs, allow_jumps=self._in_loop)
        return ElseContext(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        if exc_type is not None:
            self._circuit._pop_scope()
            return False
        true_block = self._circuit._pop_scope()
        if self._in_loop:
            operation = IfElsePlaceholder(self._condition, true_block, label=self._label)
            resources = operation.placeholder_resources()
            self._appended_instructions = self._circuit.append(operation, resources.qubits, resources.clbits)
        else:
            true_body = true_block.build(true_block.qubits, true_block.clbits)
            self._appended_instructions = self._circuit.append(IfElseOp(self._condition, true_body=true_body, false_body=None, label=self._label), tuple(true_body.qubits), tuple(true_body.clbits))
        return False

class ElseContext:
    """A context manager for building up an ``else`` statements onto circuits in a natural order,
    without having to construct the statement body first.

    Instances of this context manager should only ever be gained as the output of the
    :obj:`.IfContext` manager, so they know what they refer to.  Instances of this context are
    "friends" of the circuit that created the :obj:`.IfContext` that in turn created this object.
    The context will manipulate the circuit's defined scopes when it is entered (by popping the old
    :obj:`.IfElseOp` if it exists and pushing a new scope onto the stack) and exited (by popping its
    scope, building it, and appending the resulting :obj:`.IfElseOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    __slots__ = ('_if_instruction', '_if_registers', '_if_context', '_used')

    def __init__(self, if_context: IfContext):
        if False:
            print('Hello World!')
        self._if_instruction = None
        self._if_registers = None
        self._if_context = if_context
        self._used = False

    def __enter__(self):
        if False:
            while True:
                i = 10
        if self._used:
            raise CircuitError("Cannot re-use an 'else' context.")
        self._used = True
        appended_instructions = self._if_context.appended_instructions
        circuit = self._if_context.circuit
        if appended_instructions is None:
            raise CircuitError("Cannot attach an 'else' branch to an incomplete 'if' block.")
        if len(appended_instructions) != 1:
            raise CircuitError("Cannot attach an 'else' to a broadcasted 'if' block.")
        appended = appended_instructions[0]
        instruction = circuit._peek_previous_instruction_in_scope()
        if appended is not instruction:
            raise CircuitError(f"The 'if' block is not the most recent instruction in the circuit. Expected to find: {appended!r}, but instead found: {instruction!r}.")
        self._if_instruction = circuit._pop_previous_instruction_in_scope()
        if isinstance(self._if_instruction.operation, IfElseOp):
            self._if_registers = set(self._if_instruction.operation.blocks[0].cregs).union(self._if_instruction.operation.blocks[0].qregs)
        else:
            self._if_registers = self._if_instruction.operation.registers()
        circuit._push_scope(self._if_instruction.qubits, self._if_instruction.clbits, registers=self._if_registers, allow_jumps=self._if_context.in_loop)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        circuit = self._if_context.circuit
        if exc_type is not None:
            circuit._pop_scope()
            circuit._append(self._if_instruction)
            self._used = False
            return False
        false_block = circuit._pop_scope()
        if isinstance(self._if_instruction.operation, IfElsePlaceholder):
            if_operation = self._if_instruction.operation.with_false_block(false_block)
            resources = if_operation.placeholder_resources()
            circuit.append(if_operation, resources.qubits, resources.clbits)
        else:
            true_body = self._if_instruction.operation.blocks[0]
            false_body = false_block.build(false_block.qubits, false_block.clbits)
            (true_body, false_body) = unify_circuit_resources((true_body, false_body))
            circuit.append(IfElseOp(self._if_context.condition, true_body, false_body, label=self._if_instruction.operation.label), tuple(true_body.qubits), tuple(true_body.clbits))
        return False