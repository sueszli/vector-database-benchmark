"""
Circuit simulation for the CNOTDihedral class
"""
from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay

def _append_circuit(elem, circuit, qargs=None):
    if False:
        i = 10
        return i + 15
    'Update a CNOTDihedral element inplace by applying a CNOTDihedral circuit.\n\n    Args:\n        elem (CNOTDihedral): the CNOTDihedral element to update.\n        circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.\n        qargs (list or None): The qubits to apply gates to.\n    Returns:\n        CNOTDihedral: the updated CNOTDihedral.\n    Raises:\n        QiskitError: if input gates cannot be decomposed into CNOTDihedral gates.\n    '
    if qargs is None:
        qargs = list(range(elem.num_qubits))
    if isinstance(circuit, (Barrier, Delay)):
        return elem
    if isinstance(circuit, QuantumCircuit):
        gate = circuit.to_instruction()
    else:
        gate = circuit
    if gate.name == 'cx':
        if len(qargs) != 2:
            raise QiskitError('Invalid qubits for 2-qubit gate cx.')
        elem._append_cx(qargs[0], qargs[1])
        return elem
    elif gate.name == 'cz':
        if len(qargs) != 2:
            raise QiskitError('Invalid qubits for 2-qubit gate cz.')
        elem._append_phase(7, qargs[1])
        elem._append_phase(7, qargs[0])
        elem._append_cx(qargs[1], qargs[0])
        elem._append_phase(2, qargs[0])
        elem._append_cx(qargs[1], qargs[0])
        elem._append_phase(7, qargs[1])
        elem._append_phase(7, qargs[0])
        return elem
    if gate.name == 'ccz':
        if len(qargs) != 3:
            raise QiskitError('Invalid qubits for 2-qubit gate cx.')
        elem._append_cx(qargs[1], qargs[2])
        elem._append_phase(7, qargs[2])
        elem._append_cx(qargs[0], qargs[2])
        elem._append_phase(1, qargs[2])
        elem._append_cx(qargs[1], qargs[2])
        elem._append_phase(1, qargs[1])
        elem._append_phase(7, qargs[2])
        elem._append_cx(qargs[0], qargs[2])
        elem._append_cx(qargs[0], qargs[1])
        elem._append_phase(1, qargs[2])
        elem._append_phase(1, qargs[0])
        elem._append_phase(7, qargs[1])
        elem._append_cx(qargs[0], qargs[1])
        return elem
    if gate.name == 'id':
        if len(qargs) != 1:
            raise QiskitError('Invalid qubits for 1-qubit gate id.')
        return elem
    if gate.definition is None:
        raise QiskitError(f'Cannot apply Instruction: {gate.name}')
    if not isinstance(gate.definition, QuantumCircuit):
        raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(gate.name, type(gate.definition)))
    flat_instr = gate.definition
    bit_indices = {bit: index for bits in [flat_instr.qubits, flat_instr.clbits] for (index, bit) in enumerate(bits)}
    for instruction in gate.definition:
        if isinstance(instruction.operation, (Barrier, Delay)):
            continue
        new_qubits = [qargs[bit_indices[tup]] for tup in instruction.qubits]
        if instruction.operation.name == 'x' or gate.name == 'x':
            if len(new_qubits) != 1:
                raise QiskitError('Invalid qubits for 1-qubit gate x.')
            elem._append_x(new_qubits[0])
        elif instruction.operation.name == 'z' or gate.name == 'z':
            if len(new_qubits) != 1:
                raise QiskitError('Invalid qubits for 1-qubit gate z.')
            elem._append_phase(4, new_qubits[0])
        elif instruction.operation.name == 'y' or gate.name == 'y':
            if len(new_qubits) != 1:
                raise QiskitError('Invalid qubits for 1-qubit gate y.')
            elem._append_x(new_qubits[0])
            elem._append_phase(4, new_qubits[0])
        elif instruction.operation.name == 'p' or gate.name == 'p':
            if len(new_qubits) != 1 or len(instruction.operation.params) != 1:
                raise QiskitError('Invalid qubits or params for 1-qubit gate p.')
            elem._append_phase(int(4 * instruction.operation.params[0] / np.pi), new_qubits[0])
        elif instruction.operation.name == 't' or gate.name == 't':
            if len(new_qubits) != 1:
                raise QiskitError('Invalid qubits for 1-qubit gate t.')
            elem._append_phase(1, new_qubits[0])
        elif instruction.operation.name == 'tdg' or gate.name == 'tdg':
            if len(new_qubits) != 1:
                raise QiskitError('Invalid qubits for 1-qubit gate tdg.')
            elem._append_phase(7, new_qubits[0])
        elif instruction.operation.name == 's' or gate.name == 's':
            if len(new_qubits) != 1:
                raise QiskitError('Invalid qubits for 1-qubit gate s.')
            elem._append_phase(2, new_qubits[0])
        elif instruction.operation.name == 'sdg' or gate.name == 'sdg':
            if len(new_qubits) != 1:
                raise QiskitError('Invalid qubits for 1-qubit gate sdg.')
            elem._append_phase(6, new_qubits[0])
        elif instruction.operation.name == 'cx':
            if len(new_qubits) != 2:
                raise QiskitError('Invalid qubits for 2-qubit gate cx.')
            elem._append_cx(new_qubits[0], new_qubits[1])
        elif instruction.operation.name == 'cz':
            if len(new_qubits) != 2:
                raise QiskitError('Invalid qubits for 2-qubit gate cz.')
            elem._append_phase(7, new_qubits[1])
            elem._append_phase(7, new_qubits[0])
            elem._append_cx(new_qubits[1], new_qubits[0])
            elem._append_phase(2, new_qubits[0])
            elem._append_cx(new_qubits[1], new_qubits[0])
            elem._append_phase(7, new_qubits[1])
            elem._append_phase(7, new_qubits[0])
        elif instruction.operation.name == 'cs' or gate.name == 'cs':
            if len(new_qubits) != 2:
                raise QiskitError('Invalid qubits for 2-qubit gate cs.')
            elem._append_phase(1, new_qubits[1])
            elem._append_phase(1, new_qubits[0])
            elem._append_cx(new_qubits[1], new_qubits[0])
            elem._append_phase(7, new_qubits[0])
            elem._append_cx(new_qubits[1], new_qubits[0])
        elif instruction.operation.name == 'csdg' or gate.name == 'csdg':
            if len(new_qubits) != 2:
                raise QiskitError('Invalid qubits for 2-qubit gate csdg.')
            elem._append_phase(7, new_qubits[1])
            elem._append_phase(7, new_qubits[0])
            elem._append_cx(new_qubits[1], new_qubits[0])
            elem._append_phase(1, new_qubits[0])
            elem._append_cx(new_qubits[1], new_qubits[0])
        elif instruction.operation.name == 'swap' or gate.name == 'swap':
            if len(new_qubits) != 2:
                raise QiskitError('Invalid qubits for 2-qubit gate swap.')
            elem._append_cx(new_qubits[0], new_qubits[1])
            elem._append_cx(new_qubits[1], new_qubits[0])
            elem._append_cx(new_qubits[0], new_qubits[1])
        elif instruction.operation.name == 'ccz':
            if len(new_qubits) != 3:
                raise QiskitError('Invalid qubits for 3-qubit gate ccz.')
            elem._append_cx(new_qubits[1], new_qubits[2])
            elem._append_phase(7, new_qubits[2])
            elem._append_cx(new_qubits[0], new_qubits[2])
            elem._append_phase(1, new_qubits[2])
            elem._append_cx(new_qubits[1], new_qubits[2])
            elem._append_phase(1, new_qubits[1])
            elem._append_phase(7, new_qubits[2])
            elem._append_cx(new_qubits[0], new_qubits[2])
            elem._append_cx(new_qubits[0], new_qubits[1])
            elem._append_phase(1, new_qubits[2])
            elem._append_phase(1, new_qubits[0])
            elem._append_phase(7, new_qubits[1])
            elem._append_cx(new_qubits[0], new_qubits[1])
        elif instruction.operation.name == 'id':
            pass
        else:
            raise QiskitError(f'Not a CNOT-Dihedral gate: {instruction.operation.name}')
    return elem