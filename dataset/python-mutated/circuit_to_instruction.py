"""Helper function for converting a circuit to an instruction."""
from qiskit.exceptions import QiskitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit

def circuit_to_instruction(circuit, parameter_map=None, equivalence_library=None, label=None):
    if False:
        i = 10
        return i + 15
    "Build an :class:`~.circuit.Instruction` object from a :class:`.QuantumCircuit`.\n\n    The instruction is anonymous (not tied to a named quantum register),\n    and so can be inserted into another circuit. The instruction will\n    have the same string name as the circuit.\n\n    Args:\n        circuit (QuantumCircuit): the input circuit.\n        parameter_map (dict): For parameterized circuits, a mapping from\n           parameters in the circuit to parameters to be used in the instruction.\n           If None, existing circuit parameters will also parameterize the\n           instruction.\n        equivalence_library (EquivalenceLibrary): Optional equivalence library\n           where the converted instruction will be registered.\n        label (str): Optional instruction label.\n\n    Raises:\n        QiskitError: if parameter_map is not compatible with circuit\n\n    Return:\n        qiskit.circuit.Instruction: an instruction equivalent to the action of the\n        input circuit. Upon decomposition, this instruction will\n        yield the components comprising the original circuit.\n\n    Example:\n        .. code-block::\n\n            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit\n            from qiskit.converters import circuit_to_instruction\n\n            q = QuantumRegister(3, 'q')\n            c = ClassicalRegister(3, 'c')\n            circ = QuantumCircuit(q, c)\n            circ.h(q[0])\n            circ.cx(q[0], q[1])\n            circ.measure(q[0], c[0])\n            circ.rz(0.5, q[1]).c_if(c, 2)\n            circuit_to_instruction(circ)\n    "
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    if parameter_map is None:
        parameter_dict = {p: p for p in circuit.parameters}
    else:
        parameter_dict = circuit._unroll_param_dict(parameter_map)
    if parameter_dict.keys() != circuit.parameters:
        raise QiskitError('parameter_map should map all circuit parameters. Circuit parameters: {}, parameter_map: {}'.format(circuit.parameters, parameter_dict))
    out_instruction = Instruction(name=circuit.name, num_qubits=circuit.num_qubits, num_clbits=circuit.num_clbits, params=[*parameter_dict.values()], label=label)
    out_instruction._condition = None
    target = circuit.assign_parameters(parameter_dict, inplace=False)
    if equivalence_library is not None:
        equivalence_library.add_equivalence(out_instruction, target)
    regs = []
    if out_instruction.num_qubits > 0:
        q = QuantumRegister(out_instruction.num_qubits, 'q')
        regs.append(q)
    if out_instruction.num_clbits > 0:
        c = ClassicalRegister(out_instruction.num_clbits, 'c')
        regs.append(c)
    qubit_map = {bit: q[idx] for (idx, bit) in enumerate(circuit.qubits)}
    clbit_map = {bit: c[idx] for (idx, bit) in enumerate(circuit.clbits)}
    definition = [instruction.replace(qubits=[qubit_map[y] for y in instruction.qubits], clbits=[clbit_map[y] for y in instruction.clbits]) for instruction in target.data]
    for rule in definition:
        condition = getattr(rule.operation, 'condition', None)
        if condition:
            (reg, val) = condition
            if isinstance(reg, Clbit):
                rule.operation = rule.operation.c_if(clbit_map[reg], val)
            elif reg.size == c.size:
                rule.operation = rule.operation.c_if(c, val)
            else:
                raise QiskitError('Cannot convert condition in circuit with multiple classical registers to instruction')
    qc = QuantumCircuit(*regs, name=out_instruction.name)
    for instruction in definition:
        qc._append(instruction)
    if circuit.global_phase:
        qc.global_phase = circuit.global_phase
    out_instruction.definition = qc
    return out_instruction