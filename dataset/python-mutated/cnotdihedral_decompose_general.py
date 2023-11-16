"""
Circuit synthesis for the CNOTDihedral class.
"""
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit

def synth_cnotdihedral_general(elem):
    if False:
        for i in range(10):
            print('nop')
    'Decompose a CNOTDihedral element into a QuantumCircuit.\n\n    Decompose a general CNOTDihedral elements.\n    The number of CNOT gates is not necessarily optimal.\n    For a decomposition of a 1-qubit or 2-qubit element, call\n    synth_cnotdihedral_two_qubits.\n\n    Args:\n        elem (CNOTDihedral): a CNOTDihedral element.\n\n    Return:\n        QuantumCircuit: a circuit implementation of the CNOTDihedral element.\n\n    Raises:\n        QiskitError: if the element could not be decomposed into a circuit.\n\n    Reference:\n        1. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,\n           *Scalable randomised benchmarking of non-Clifford gates*,\n           npj Quantum Inf 2, 16012 (2016).\n    '
    num_qubits = elem.num_qubits
    circuit = QuantumCircuit(num_qubits)
    elem_cpy = elem.copy()
    if not np.allclose(np.linalg.det(elem_cpy.linear) % 2, 1):
        raise QiskitError('Linear part is not invertible.')
    for i in range(num_qubits):
        if elem.shift[i]:
            circuit.x(i)
            elem_cpy._append_x(i)
    for i in range(num_qubits):
        if not elem_cpy.linear[i][i]:
            for j in range(i + 1, num_qubits):
                if elem_cpy.linear[j][i]:
                    circuit.cx(j, i)
                    circuit.cx(i, j)
                    circuit.cx(j, i)
                    elem_cpy._append_cx(j, i)
                    elem_cpy._append_cx(i, j)
                    elem_cpy._append_cx(j, i)
                    break
        for j in range(num_qubits):
            if j != i:
                if elem_cpy.linear[j][i]:
                    circuit.cx(i, j)
                    elem_cpy._append_cx(i, j)
    if not (elem_cpy.shift == np.zeros(num_qubits)).all() or not (elem_cpy.linear == np.eye(num_qubits)).all():
        raise QiskitError('Cannot do Gauss elimination on linear part.')
    new_elem = elem_cpy.copy()
    new_elem.poly.weight_0 = 0
    new_elem.poly.weight_1 = np.zeros(num_qubits, dtype=np.int8)
    new_elem.poly.weight_2 = np.zeros(int(num_qubits * (num_qubits - 1) / 2), dtype=np.int8)
    new_elem.poly.weight_3 = np.zeros(int(num_qubits * (num_qubits - 1) * (num_qubits - 2) / 6), dtype=np.int8)
    new_circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            for k in range(j + 1, num_qubits):
                if elem_cpy.poly.get_term([i, j, k]) != 0:
                    new_elem._append_cx(i, k)
                    new_elem._append_cx(j, k)
                    new_elem._append_phase(1, k)
                    new_elem._append_cx(i, k)
                    new_elem._append_cx(j, k)
                    new_circuit.cx(i, k)
                    new_circuit.cx(j, k)
                    new_circuit.p(np.pi / 4, [k])
                    new_circuit.cx(i, k)
                    new_circuit.cx(j, k)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            tpow1 = elem_cpy.poly.get_term([i, j])
            tpow2 = new_elem.poly.get_term([i, j])
            tpow = (tpow2 - tpow1) / 2 % 4
            if tpow != 0:
                new_elem._append_cx(i, j)
                new_elem._append_phase(tpow, j)
                new_elem._append_cx(i, j)
                new_circuit.cx(i, j)
                new_circuit.p(tpow * np.pi / 4, [j])
                new_circuit.cx(i, j)
    for i in range(num_qubits):
        tpow1 = elem_cpy.poly.get_term([i])
        tpow2 = new_elem.poly.get_term([i])
        tpow = (tpow1 - tpow2) % 8
        if tpow != 0:
            new_elem._append_phase(tpow, i)
            new_circuit.p(tpow * np.pi / 4, [i])
    if elem.poly != new_elem.poly:
        raise QiskitError('Could not recover phase polynomial.')
    inv_circuit = circuit.inverse()
    return new_circuit.compose(inv_circuit)