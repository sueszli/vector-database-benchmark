"""Utility methods for canonicalising various Qiskit objects, to help with testing."""
import threading
from qiskit.circuit import BreakLoopOp, CircuitInstruction, ContinueLoopOp, ControlFlowOp, ForLoopOp, Parameter, QuantumCircuit

class _CanonicalParametersIterator:
    """An object that, when iterated through, will produce the same sequence of parameters as every
    other instance of this iterator."""
    __parameters = []
    __mutex = threading.Lock()

    def __init__(self):
        if False:
            return 10
        self._counter = 0

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        with self.__mutex:
            if len(self.__parameters) >= self._counter:
                param = Parameter(f'_canonicalization_loop_{self._counter}')
                self.__parameters.append(param)
            out = self.__parameters[self._counter]
            self._counter += 1
            return out

def canonicalize_control_flow(circuit: QuantumCircuit) -> QuantumCircuit:
    if False:
        i = 10
        return i + 15
    'Canonicalize all control-flow operations in a circuit.\n\n    This is not an efficient operation, and does not affect any properties of the circuit.  Its\n    intent is to normalise parts of circuits that have a non-deterministic construction.  These are\n    the ordering of bit arguments in control-flow blocks output by the builder interface, and\n    automatically generated ``for``-loop variables.\n\n    The canonical form sorts the bits in the arguments of these operations so that they always\n    appear in the order they were originally added to the outer-most circuit.  For-loop variables\n    are re-bound into new, cached auto-generated ones.'
    params = iter(_CanonicalParametersIterator())
    base_bit_order = {bit: i for (i, bit) in enumerate(circuit.qubits)}
    base_bit_order.update(((bit, i) for (i, bit) in enumerate(circuit.clbits)))

    def worker(circuit, bit_map=None):
        if False:
            while True:
                i = 10
        if bit_map is None:
            bit_map = {bit: bit for bits in (circuit.qubits, circuit.clbits) for bit in bits}

        def bit_key(bit):
            if False:
                print('Hello World!')
            return base_bit_order[bit_map[bit]]
        out = QuantumCircuit(sorted(circuit.qubits, key=bit_key), sorted(circuit.clbits, key=bit_key), *circuit.qregs, *circuit.cregs, name=circuit.name, global_phase=circuit.global_phase, metadata=circuit.metadata)
        for instruction in circuit.data:
            new_instruction = instruction
            if isinstance(new_instruction.operation, ControlFlowOp):
                op = new_instruction.operation
                first_block = op.blocks[0]
                inner_bit_map = dict(zip(first_block.qubits, (bit_map[bit] for bit in new_instruction.qubits)))
                inner_bit_map.update(zip(first_block.clbits, (bit_map[bit] for bit in new_instruction.clbits)))
                new_instruction = CircuitInstruction(operation=op.replace_blocks([worker(block, inner_bit_map) for block in op.blocks]), qubits=sorted(new_instruction.qubits, key=bit_key), clbits=sorted(new_instruction.clbits, key=bit_key))
            elif isinstance(new_instruction.operation, (BreakLoopOp, ContinueLoopOp)):
                new_instruction = new_instruction.replace(qubits=sorted(new_instruction.qubits, key=bit_key), clbits=sorted(new_instruction.clbits, key=bit_key))
            if isinstance(new_instruction.operation, ForLoopOp):
                old_op = new_instruction.operation
                (indexset, loop_param, body) = old_op.params
                if loop_param is not None:
                    new_loop_param = next(params)
                    new_op = ForLoopOp(indexset, new_loop_param, body.assign_parameters({loop_param: new_loop_param}))
                    new_instruction = new_instruction.replace(operation=new_op)
            out._append(new_instruction)
        return out
    return worker(circuit)