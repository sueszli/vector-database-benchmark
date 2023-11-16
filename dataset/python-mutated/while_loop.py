"""Circuit operation representing a ``while`` loop."""
from __future__ import annotations
from qiskit.circuit import Clbit, ClassicalRegister, QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.circuit.exceptions import CircuitError
from ._builder_utils import validate_condition, condition_resources
from .control_flow import ControlFlowOp

class WhileLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit (``body``) until
    a condition (``condition``) evaluates as False.

    Parameters:
        condition: A condition to be checked prior to executing ``body``. Can be
            specified as either a tuple of a ``ClassicalRegister`` to be tested
            for equality with a given ``int``, or as a tuple of a ``Clbit`` to
            be compared to either a ``bool`` or an ``int``.
        body: The loop body to be repeatedly executed.
        label: An optional label for identifying the instruction.

    The classical bits used in ``condition`` must be a subset of those attached
    to ``body``.

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────────────┐
        q_0: ┤0            ├
             │             │
        q_1: ┤1            ├
             │  while_loop │
        q_2: ┤2            ├
             │             │
        c_0: ╡0            ╞
             └─────────────┘

    """

    def __init__(self, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, body: QuantumCircuit, label: str | None=None):
        if False:
            print('Hello World!')
        num_qubits = body.num_qubits
        num_clbits = body.num_clbits
        super().__init__('while_loop', num_qubits, num_clbits, [body], label=label)
        self.condition = validate_condition(condition)

    @property
    def params(self):
        if False:
            print('Hello World!')
        return self._params

    @params.setter
    def params(self, parameters):
        if False:
            while True:
                i = 10
        (body,) = parameters
        if not isinstance(body, QuantumCircuit):
            raise CircuitError(f'WhileLoopOp expects a body parameter of type QuantumCircuit, but received {type(body)}.')
        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(f'Attempted to assign a body parameter with a num_qubits or num_clbits different than that of the WhileLoopOp. WhileLoopOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}.')
        self._params = [body]

    @property
    def blocks(self):
        if False:
            i = 10
            return i + 15
        return (self._params[0],)

    def replace_blocks(self, blocks):
        if False:
            for i in range(10):
                print('nop')
        (body,) = blocks
        return WhileLoopOp(self.condition, body, label=self.label)

    def c_if(self, classical, val):
        if False:
            return 10
        raise NotImplementedError('WhileLoopOp cannot be classically controlled through Instruction.c_if. Please use an IfElseOp instead.')

class WhileLoopContext:
    """A context manager for building up while loops onto circuits in a natural order, without
    having to construct the loop body first.

    Within the block, a lot of the bookkeeping is done for you; you do not need to keep track of
    which qubits and clbits you are using, for example.  All normal methods of accessing the qubits
    on the underlying :obj:`~QuantumCircuit` will work correctly, and resolve into correct accesses
    within the interior block.

    You generally should never need to instantiate this object directly.  Instead, use
    :obj:`.QuantumCircuit.while_loop` in its context-manager form, i.e. by not supplying a ``body``
    or sets of qubits and clbits.

    Example usage::

        from qiskit.circuit import QuantumCircuit, Clbit, Qubit
        bits = [Qubit(), Qubit(), Clbit()]
        qc = QuantumCircuit(bits)

        with qc.while_loop((bits[2], 0)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    __slots__ = ('_circuit', '_condition', '_label')

    def __init__(self, circuit: QuantumCircuit, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, *, label: str | None=None):
        if False:
            for i in range(10):
                print('nop')
        self._circuit = circuit
        self._condition = validate_condition(condition)
        self._label = label

    def __enter__(self):
        if False:
            while True:
                i = 10
        resources = condition_resources(self._condition)
        self._circuit._push_scope(clbits=resources.clbits, registers=resources.cregs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        if exc_type is not None:
            self._circuit._pop_scope()
            return False
        scope = self._circuit._pop_scope()
        body = scope.build(scope.qubits, scope.clbits)
        self._circuit.append(WhileLoopOp(self._condition, body, label=self._label), body.qubits, body.clbits)
        return False