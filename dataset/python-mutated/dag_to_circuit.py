"""Helper function for converting a dag to a circuit."""
import copy
from qiskit.circuit import QuantumCircuit, CircuitInstruction

def dag_to_circuit(dag, copy_operations=True):
    if False:
        print('Hello World!')
    "Build a ``QuantumCircuit`` object from a ``DAGCircuit``.\n\n    Args:\n        dag (DAGCircuit): the input dag.\n        copy_operations (bool): Deep copy the operation objects\n            in the :class:`~.DAGCircuit` for the output :class:`~.QuantumCircuit`.\n            This should only be set to ``False`` if the input :class:`~.DAGCircuit`\n            will not be used anymore as the operations in the output\n            :class:`~.QuantumCircuit` will be shared instances and\n            modifications to operations in the :class:`~.DAGCircuit` will\n            be reflected in the :class:`~.QuantumCircuit` (and vice versa).\n\n    Return:\n        QuantumCircuit: the circuit representing the input dag.\n\n    Example:\n        .. plot::\n           :include-source:\n\n           from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit\n           from qiskit.dagcircuit import DAGCircuit\n           from qiskit.converters import circuit_to_dag\n           from qiskit.circuit.library.standard_gates import CHGate, U2Gate, CXGate\n           from qiskit.converters import dag_to_circuit\n\n           q = QuantumRegister(3, 'q')\n           c = ClassicalRegister(3, 'c')\n           circ = QuantumCircuit(q, c)\n           circ.h(q[0])\n           circ.cx(q[0], q[1])\n           circ.measure(q[0], c[0])\n           circ.rz(0.5, q[1]).c_if(c, 2)\n           dag = circuit_to_dag(circ)\n           circuit = dag_to_circuit(dag)\n           circuit.draw('mpl')\n    "
    name = dag.name or None
    circuit = QuantumCircuit(dag.qubits, dag.clbits, *dag.qregs.values(), *dag.cregs.values(), name=name, global_phase=dag.global_phase)
    circuit.metadata = dag.metadata
    circuit.calibrations = dag.calibrations
    for node in dag.topological_op_nodes():
        op = node.op
        if copy_operations:
            op = copy.deepcopy(op)
        circuit._append(CircuitInstruction(op, node.qargs, node.cargs))
    circuit.duration = dag.duration
    circuit.unit = dag.unit
    return circuit