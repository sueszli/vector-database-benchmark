"""Helper function for converting a circuit to a dag"""
import copy
from qiskit.dagcircuit.dagcircuit import DAGCircuit

def circuit_to_dag(circuit, copy_operations=True, *, qubit_order=None, clbit_order=None):
    if False:
        while True:
            i = 10
    "Build a :class:`.DAGCircuit` object from a :class:`.QuantumCircuit`.\n\n    Args:\n        circuit (QuantumCircuit): the input circuit.\n        copy_operations (bool): Deep copy the operation objects\n            in the :class:`~.QuantumCircuit` for the output :class:`~.DAGCircuit`.\n            This should only be set to ``False`` if the input :class:`~.QuantumCircuit`\n            will not be used anymore as the operations in the output\n            :class:`~.DAGCircuit` will be shared instances and modifications to\n            operations in the :class:`~.DAGCircuit` will be reflected in the\n            :class:`~.QuantumCircuit` (and vice versa).\n        qubit_order (Iterable[~qiskit.circuit.Qubit] or None): the order that the qubits should be\n            indexed in the output DAG.  Defaults to the same order as in the circuit.\n        clbit_order (Iterable[Clbit] or None): the order that the clbits should be indexed in the\n            output DAG.  Defaults to the same order as in the circuit.\n\n    Return:\n        DAGCircuit: the DAG representing the input circuit.\n\n    Raises:\n        ValueError: if the ``qubit_order`` or ``clbit_order`` parameters do not match the bits in\n            the circuit.\n\n    Example:\n        .. code-block::\n\n            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit\n            from qiskit.dagcircuit import DAGCircuit\n            from qiskit.converters import circuit_to_dag\n\n            q = QuantumRegister(3, 'q')\n            c = ClassicalRegister(3, 'c')\n            circ = QuantumCircuit(q, c)\n            circ.h(q[0])\n            circ.cx(q[0], q[1])\n            circ.measure(q[0], c[0])\n            circ.rz(0.5, q[1]).c_if(c, 2)\n            dag = circuit_to_dag(circ)\n    "
    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata
    if qubit_order is None:
        qubits = circuit.qubits
    elif len(qubit_order) != circuit.num_qubits or set(qubit_order) != set(circuit.qubits):
        raise ValueError("'qubit_order' does not contain exactly the same qubits as the circuit")
    else:
        qubits = qubit_order
    if clbit_order is None:
        clbits = circuit.clbits
    elif len(clbit_order) != circuit.num_clbits or set(clbit_order) != set(circuit.clbits):
        raise ValueError("'clbit_order' does not contain exactly the same clbits as the circuit")
    else:
        clbits = clbit_order
    dagcircuit.add_qubits(qubits)
    dagcircuit.add_clbits(clbits)
    for register in circuit.qregs:
        dagcircuit.add_qreg(register)
    for register in circuit.cregs:
        dagcircuit.add_creg(register)
    for instruction in circuit.data:
        op = instruction.operation
        if copy_operations:
            op = copy.deepcopy(op)
        dagcircuit.apply_operation_back(op, instruction.qubits, instruction.clbits, check=False)
    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit