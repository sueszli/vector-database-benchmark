"""
PauliList utility functions.
"""
from __future__ import annotations
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList

def pauli_basis(num_qubits: int, weight: bool=False) -> PauliList:
    if False:
        for i in range(10):
            print('nop')
    'Return the ordered PauliList for the n-qubit Pauli basis.\n\n    Args:\n        num_qubits (int): number of qubits\n        weight (bool): if True optionally return the basis sorted by Pauli weight\n                       rather than lexicographic order (Default: False)\n\n    Returns:\n        PauliList: the Paulis for the basis\n    '
    pauli_1q = PauliList(['I', 'X', 'Y', 'Z'])
    if num_qubits == 1:
        return pauli_1q
    pauli = pauli_1q
    for _ in range(num_qubits - 1):
        pauli = pauli_1q.tensor(pauli)
    if weight:
        return pauli.sort(weight=True)
    return pauli