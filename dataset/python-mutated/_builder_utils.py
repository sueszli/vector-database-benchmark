"""Private utility functions that are used by the builder interfaces."""
from __future__ import annotations
import dataclasses
from typing import Iterable, Tuple, Set, Union, TypeVar
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import QuantumRegister
_ConditionT = TypeVar('_ConditionT', bound=Union[Tuple[ClassicalRegister, int], Tuple[Clbit, int], expr.Expr])

def validate_condition(condition: _ConditionT) -> _ConditionT:
    if False:
        print('Hello World!')
    'Validate that a condition is in a valid format and return it, but raise if it is invalid.\n\n    Args:\n        condition: the condition to be tested for validity.  Must be either the legacy 2-tuple\n            format, or a :class:`~.expr.Expr` that has `Bool` type.\n\n    Raises:\n        CircuitError: if the condition is not in a valid format.\n\n    Returns:\n        The same condition as passed, if it was valid.\n    '
    if isinstance(condition, expr.Expr):
        if condition.type.kind is not types.Bool:
            raise CircuitError(f"Classical conditions must be expressions with the type 'Bool()', not '{condition.type}'.")
        return condition
    try:
        (bits, value) = condition
        if isinstance(bits, (ClassicalRegister, Clbit)) and isinstance(value, int):
            return (bits, value)
    except (TypeError, ValueError):
        pass
    raise CircuitError(f"A classical condition should be a 2-tuple of `(ClassicalRegister | Clbit, int)`, but received '{condition!r}'.")

@dataclasses.dataclass
class LegacyResources:
    """A pair of the :class:`.Clbit` and :class:`.ClassicalRegister` resources used by some other
    object (such as a legacy condition or :class:`.expr.Expr` node)."""
    clbits: tuple[Clbit, ...]
    cregs: tuple[ClassicalRegister, ...]

def node_resources(node: expr.Expr) -> LegacyResources:
    if False:
        while True:
            i = 10
    'Get the legacy classical resources (:class:`.Clbit` and :class:`.ClassicalRegister`)\n    referenced by an :class:`~.expr.Expr`.'
    clbits = {}
    cregs = {}
    for var in expr.iter_vars(node):
        if isinstance(var.var, Clbit):
            clbits[var.var] = None
        elif isinstance(var.var, ClassicalRegister):
            clbits.update(((bit, None) for bit in var.var))
            cregs[var.var] = None
    return LegacyResources(tuple(clbits), tuple(cregs))

def condition_resources(condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr) -> LegacyResources:
    if False:
        return 10
    'Get the legacy classical resources (:class:`.Clbit` and :class:`.ClassicalRegister`)\n    referenced by a legacy condition or an :class:`~.expr.Expr`.'
    if isinstance(condition, expr.Expr):
        return node_resources(condition)
    (target, _) = condition
    if isinstance(target, ClassicalRegister):
        return LegacyResources(tuple(target), (target,))
    return LegacyResources((target,), ())

def partition_registers(registers: Iterable[Register]) -> Tuple[Set[QuantumRegister], Set[ClassicalRegister]]:
    if False:
        return 10
    'Partition a sequence of registers into its quantum and classical registers.'
    qregs = set()
    cregs = set()
    for register in registers:
        if isinstance(register, QuantumRegister):
            qregs.add(register)
        elif isinstance(register, ClassicalRegister):
            cregs.add(register)
        else:
            raise CircuitError(f'Unknown register: {register}.')
    return (qregs, cregs)

def unify_circuit_resources(circuits: Iterable[QuantumCircuit]) -> Iterable[QuantumCircuit]:
    if False:
        while True:
            i = 10
    "\n    Ensure that all the given ``circuits`` have all the same qubits, clbits and registers, and\n    that they are defined in the same order.  The order is important for binding when the bodies are\n    used in the 3-tuple :obj:`.Instruction` context.\n\n    This function will preferentially try to mutate its inputs if they share an ordering, but if\n    not, it will rebuild two new circuits.  This is to avoid coupling too tightly to the inner\n    class; there is no real support for deleting or re-ordering bits within a :obj:`.QuantumCircuit`\n    context, and we don't want to rely on the *current* behaviour of the private APIs, since they\n    are very liable to change.  No matter the method used, circuits with unified bits and registers\n    are returned.\n    "
    circuits = tuple(circuits)
    if len(circuits) < 2:
        return circuits
    qubits = []
    clbits = []
    for circuit in circuits:
        if circuit.qubits[:len(qubits)] != qubits:
            return _unify_circuit_resources_rebuild(circuits)
        if circuit.clbits[:len(qubits)] != clbits:
            return _unify_circuit_resources_rebuild(circuits)
        if circuit.num_qubits > len(qubits):
            qubits = list(circuit.qubits)
        if circuit.num_clbits > len(clbits):
            clbits = list(circuit.clbits)
    for circuit in circuits:
        circuit.add_bits(qubits[circuit.num_qubits:])
        circuit.add_bits(clbits[circuit.num_clbits:])
    return _unify_circuit_registers(circuits)

def _unify_circuit_resources_rebuild(circuits: Tuple[QuantumCircuit, ...]) -> Tuple[QuantumCircuit, QuantumCircuit]:
    if False:
        print('Hello World!')
    '\n    Ensure that all the given circuits have all the same qubits and clbits, and that they\n    are defined in the same order.  The order is important for binding when the bodies are used in\n    the 3-tuple :obj:`.Instruction` context.\n\n    This function will always rebuild the objects into new :class:`.QuantumCircuit` instances.\n    '
    (qubits, clbits) = (set(), set())
    for circuit in circuits:
        qubits.update(circuit.qubits)
        clbits.update(circuit.clbits)
    (qubits, clbits) = (list(qubits), list(clbits))
    out_circuits = []
    for circuit in circuits:
        out = QuantumCircuit(qubits, clbits, *circuit.qregs, *circuit.cregs, global_phase=circuit.global_phase)
        for instruction in circuit.data:
            out._append(instruction)
        out_circuits.append(out)
    return _unify_circuit_registers(out_circuits)

def _unify_circuit_registers(circuits: Iterable[QuantumCircuit]) -> Iterable[QuantumCircuit]:
    if False:
        i = 10
        return i + 15
    '\n    Ensure that ``true_body`` and ``false_body`` have the same registers defined within them.  These\n    do not need to be in the same order between circuits.  The two input circuits are returned,\n    mutated to have the same registers.\n    '
    circuits = tuple(circuits)
    total_registers = set()
    for circuit in circuits:
        total_registers.update(circuit.qregs)
        total_registers.update(circuit.cregs)
    for circuit in circuits:
        for register in total_registers - set(circuit.qregs) - set(circuit.cregs):
            circuit.add_register(register)
    return circuits