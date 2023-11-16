"""Shared helper utility for mapping classical resources from one circuit or DAG to another."""
from __future__ import annotations
import typing
from .bit import Bit
from .classical import expr
from .classicalregister import ClassicalRegister, Clbit

class VariableMapper(expr.ExprVisitor[expr.Expr]):
    """Stateful helper class that manages the mapping of variables in conditions and expressions.

    This is designed to be used by both :class:`.QuantumCircuit` and :class:`.DAGCircuit` when
    managing operations that need to map classical resources from one circuit to another.

    The general usage is to initialise this at the start of a many-block mapping operation, then
    call its :meth:`map_condition`, :meth:`map_target` or :meth:`map_expr` methods as appropriate,
    which will return the new object that should be used.

    If an ``add_register`` callable is given to the initialiser, the mapper will use it to attempt
    to add new aliasing registers to the outer circuit object, if there is not already a suitable
    register for the mapping available in the circuit.  If this parameter is not given, a
    ``ValueError`` will be raised instead.  The given ``add_register`` callable may choose to raise
    its own exception."""
    __slots__ = ('target_cregs', 'register_map', 'bit_map', 'add_register')

    def __init__(self, target_cregs: typing.Iterable[ClassicalRegister], bit_map: typing.Mapping[Bit, Bit], add_register: typing.Callable[[ClassicalRegister], None] | None=None):
        if False:
            while True:
                i = 10
        self.target_cregs = tuple(target_cregs)
        self.register_map = {}
        self.bit_map = bit_map
        self.add_register = add_register

    def _map_register(self, theirs: ClassicalRegister) -> ClassicalRegister:
        if False:
            while True:
                i = 10
        "Map the target's registers to suitable equivalents in the destination, adding an\n        extra one if there's no exact match."
        if (mapped_theirs := self.register_map.get(theirs.name)) is not None:
            return mapped_theirs
        mapped_bits = [self.bit_map[bit] for bit in theirs]
        for ours in self.target_cregs:
            if mapped_bits == list(ours):
                mapped_theirs = ours
                break
        else:
            if self.add_register is None:
                raise ValueError(f"Register '{theirs.name}' has no counterpart in the destination.")
            mapped_theirs = ClassicalRegister(bits=mapped_bits)
            self.add_register(mapped_theirs)
        self.register_map[theirs.name] = mapped_theirs
        return mapped_theirs

    def map_condition(self, condition, /, *, allow_reorder=False):
        if False:
            for i in range(10):
                print('nop')
        'Map the given ``condition`` so that it only references variables in the destination\n        circuit (as given to this class on initialisation).\n\n        If ``allow_reorder`` is ``True``, then when a legacy condition (the two-tuple form) is made\n        on a register that has a counterpart in the destination with all the same (mapped) bits but\n        in a different order, then that register will be used and the value suitably modified to\n        make the equality condition work.  This is maintaining legacy (tested) behaviour of\n        :meth:`.DAGCircuit.compose`; nowhere else does this, and in general this would require *far*\n        more complex classical rewriting than Terra needs to worry about in the full expression era.\n        '
        if condition is None:
            return None
        if isinstance(condition, expr.Expr):
            return self.map_expr(condition)
        (target, value) = condition
        if isinstance(target, Clbit):
            return (self.bit_map[target], value)
        if not allow_reorder:
            return (self._map_register(target), value)
        mapped_bits_order = [self.bit_map[bit] for bit in target]
        mapped_bits_set = set(mapped_bits_order)
        for register in self.target_cregs:
            if mapped_bits_set == set(register):
                mapped_theirs = register
                break
        else:
            if self.add_register is None:
                raise self.exc_type(f"Register '{target.name}' has no counterpart in the destination.")
            mapped_theirs = ClassicalRegister(bits=mapped_bits_order)
            self.add_register(mapped_theirs)
        new_order = {bit: i for (i, bit) in enumerate(mapped_bits_order)}
        value_bits = f'{value:0{len(target)}b}'[::-1]
        mapped_value = int(''.join((value_bits[new_order[bit]] for bit in mapped_theirs))[::-1], 2)
        return (mapped_theirs, mapped_value)

    def map_target(self, target, /):
        if False:
            print('Hello World!')
        'Map the runtime variables in a ``target`` of a :class:`.SwitchCaseOp` to the new circuit,\n        as defined in the ``circuit`` argument of the initialiser of this class.'
        if isinstance(target, Clbit):
            return self.bit_map[target]
        if isinstance(target, ClassicalRegister):
            return self._map_register(target)
        return self.map_expr(target)

    def map_expr(self, node: expr.Expr, /) -> expr.Expr:
        if False:
            for i in range(10):
                print('nop')
        'Map the variables in an :class:`~.expr.Expr` node to the new circuit.'
        return node.accept(self)

    def visit_var(self, node, /):
        if False:
            while True:
                i = 10
        if isinstance(node.var, Clbit):
            return expr.Var(self.bit_map[node.var], node.type)
        if isinstance(node.var, ClassicalRegister):
            return expr.Var(self._map_register(node.var), node.type)
        raise RuntimeError(f"unhandled variable in 'compose': {node}")

    def visit_value(self, node, /):
        if False:
            print('Hello World!')
        return expr.Value(node.value, node.type)

    def visit_unary(self, node, /):
        if False:
            print('Hello World!')
        return expr.Unary(node.op, node.operand.accept(self), node.type)

    def visit_binary(self, node, /):
        if False:
            i = 10
            return i + 15
        return expr.Binary(node.op, node.left.accept(self), node.right.accept(self), node.type)

    def visit_cast(self, node, /):
        if False:
            i = 10
            return i + 15
        return expr.Cast(node.operand.accept(self), node.type, implicit=node.implicit)