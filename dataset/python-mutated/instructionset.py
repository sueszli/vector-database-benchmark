"""
Instruction collection.
"""
from __future__ import annotations
from typing import Callable
from qiskit.circuit.exceptions import CircuitError
from .classicalregister import Clbit, ClassicalRegister
from .operation import Operation
from .quantumcircuitdata import CircuitInstruction

class InstructionSet:
    """Instruction collection, and their contexts."""
    __slots__ = ('_instructions', '_requester')

    def __init__(self, *, resource_requester: Callable[..., ClassicalRegister | Clbit] | None=None):
        if False:
            for i in range(10):
                print('nop')
        'New collection of instructions.\n\n        The context (``qargs`` and ``cargs`` that each instruction is attached to) is also stored\n        separately for each instruction.\n\n        Args:\n            resource_requester: A callable that takes in the classical resource used in the\n                condition, verifies that it is present in the attached circuit, resolves any indices\n                into concrete :obj:`.Clbit` instances, and returns the concrete resource.  If this\n                is not given, specifying a condition with an index is forbidden, and all concrete\n                :obj:`.Clbit` and :obj:`.ClassicalRegister` resources will be assumed to be valid.\n\n                .. note::\n\n                    The callback ``resource_requester`` is called once for each call to\n                    :meth:`.c_if`, and assumes that a call implies that the resource will now be\n                    used.  It may throw an error if the resource is not valid for usage.\n\n        '
        self._instructions: list[CircuitInstruction] = []
        self._requester = resource_requester

    def __len__(self):
        if False:
            return 10
        'Return number of instructions in set'
        return len(self._instructions)

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        'Return instruction at index'
        return self._instructions[i]

    def add(self, instruction, qargs=None, cargs=None):
        if False:
            print('Hello World!')
        'Add an instruction and its context (where it is attached).'
        if not isinstance(instruction, CircuitInstruction):
            if not isinstance(instruction, Operation):
                raise CircuitError('attempt to add non-Operation to InstructionSet')
            if qargs is None or cargs is None:
                raise CircuitError('missing qargs or cargs in old-style InstructionSet.add')
            instruction = CircuitInstruction(instruction, tuple(qargs), tuple(cargs))
        self._instructions.append(instruction)

    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Invert all instructions.'
        for (i, instruction) in enumerate(self._instructions):
            self._instructions[i] = instruction.replace(operation=instruction.operation.inverse())
        return self

    def c_if(self, classical: Clbit | ClassicalRegister | int, val: int) -> 'InstructionSet':
        if False:
            i = 10
            return i + 15
        "Set a classical equality condition on all the instructions in this set between the\n        :obj:`.ClassicalRegister` or :obj:`.Clbit` ``classical`` and value ``val``.\n\n        .. note::\n\n            This is a setter method, not an additive one.  Calling this multiple times will silently\n            override any previously set condition on any of the contained instructions; it does not\n            stack.\n\n        Args:\n            classical: the classical resource the equality condition should be on.  If this is given\n                as an integer, it will be resolved into a :obj:`.Clbit` using the same conventions\n                as the circuit these instructions are attached to.\n            val: the value the classical resource should be equal to.\n\n        Returns:\n            This same instance of :obj:`.InstructionSet`, but now mutated to have the given equality\n            condition.\n\n        Raises:\n            CircuitError: if the passed classical resource is invalid, or otherwise not resolvable\n                to a concrete resource that these instructions are permitted to access.\n\n        Example:\n            .. plot::\n               :include-source:\n\n               from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit\n\n               qr = QuantumRegister(2)\n               cr = ClassicalRegister(2)\n               qc = QuantumCircuit(qr, cr)\n               qc.h(range(2))\n               qc.measure(range(2), range(2))\n\n               # apply x gate if the classical register has the value 2 (10 in binary)\n               qc.x(0).c_if(cr, 2)\n\n               # apply y gate if bit 0 is set to 1\n               qc.y(1).c_if(0, 1)\n\n               qc.draw('mpl')\n\n        "
        if self._requester is None and (not isinstance(classical, (Clbit, ClassicalRegister))):
            raise CircuitError('Cannot pass an index as a condition variable without specifying a requester when creating this InstructionSet.')
        if self._requester is not None:
            classical = self._requester(classical)
        for instruction in self._instructions:
            instruction.operation = instruction.operation.c_if(classical, val)
        return self

    @property
    def instructions(self):
        if False:
            i = 10
            return i + 15
        'Legacy getter for the instruction components of an instruction set.  This does not\n        support mutation.'
        return [instruction.operation for instruction in self._instructions]

    @property
    def qargs(self):
        if False:
            print('Hello World!')
        'Legacy getter for the qargs components of an instruction set.  This does not support\n        mutation.'
        return [list(instruction.qubits) for instruction in self._instructions]

    @property
    def cargs(self):
        if False:
            return 10
        'Legacy getter for the cargs components of an instruction set.  This does not support\n        mutation.'
        return [list(instruction.clbits) for instruction in self._instructions]