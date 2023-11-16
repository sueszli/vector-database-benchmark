"""Helper function for converting a circuit to a gate"""
from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError

def _check_is_gate(op):
    if False:
        for i in range(10):
            print('nop')
    'Checks whether op can be converted to Gate.'
    if isinstance(op, Gate):
        return True
    elif isinstance(op, AnnotatedOperation):
        return _check_is_gate(op.base_op)
    return False

def circuit_to_gate(circuit, parameter_map=None, equivalence_library=None, label=None):
    if False:
        while True:
            i = 10
    'Build a :class:`.Gate` object from a :class:`.QuantumCircuit`.\n\n    The gate is anonymous (not tied to a named quantum register),\n    and so can be inserted into another circuit. The gate will\n    have the same string name as the circuit.\n\n    Args:\n        circuit (QuantumCircuit): the input circuit.\n        parameter_map (dict): For parameterized circuits, a mapping from\n           parameters in the circuit to parameters to be used in the gate.\n           If None, existing circuit parameters will also parameterize the\n           Gate.\n        equivalence_library (EquivalenceLibrary): Optional equivalence library\n           where the converted gate will be registered.\n        label (str): Optional gate label.\n\n    Raises:\n        QiskitError: if circuit is non-unitary or if\n            parameter_map is not compatible with circuit\n\n    Return:\n        Gate: a Gate equivalent to the action of the\n        input circuit. Upon decomposition, this gate will\n        yield the components comprising the original circuit.\n    '
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    if circuit.clbits:
        raise QiskitError('Circuit with classical bits cannot be converted to gate.')
    for instruction in circuit.data:
        if not _check_is_gate(instruction.operation):
            raise QiskitError('One or more instructions cannot be converted to a gate. "{}" is not a gate instruction'.format(instruction.operation.name))
    if parameter_map is None:
        parameter_dict = {p: p for p in circuit.parameters}
    else:
        parameter_dict = circuit._unroll_param_dict(parameter_map)
    if parameter_dict.keys() != circuit.parameters:
        raise QiskitError('parameter_map should map all circuit parameters. Circuit parameters: {}, parameter_map: {}'.format(circuit.parameters, parameter_dict))
    gate = Gate(name=circuit.name, num_qubits=circuit.num_qubits, params=[*parameter_dict.values()], label=label)
    gate.condition = None
    target = circuit.assign_parameters(parameter_dict, inplace=False)
    if equivalence_library is not None:
        equivalence_library.add_equivalence(gate, target)
    qc = QuantumCircuit(name=gate.name, global_phase=target.global_phase)
    if gate.num_qubits > 0:
        q = QuantumRegister(gate.num_qubits, 'q')
        qc.add_register(q)
    qubit_map = {bit: q[idx] for (idx, bit) in enumerate(circuit.qubits)}
    for instruction in target.data:
        qc._append(instruction.replace(qubits=tuple((qubit_map[y] for y in instruction.qubits))))
    gate.definition = qc
    return gate