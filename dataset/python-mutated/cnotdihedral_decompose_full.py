"""
Circuit synthesis for the CNOTDihedral class for all-to-all connectivity.
"""
from qiskit.synthesis.cnotdihedral.cnotdihedral_decompose_two_qubits import synth_cnotdihedral_two_qubits
from qiskit.synthesis.cnotdihedral.cnotdihedral_decompose_general import synth_cnotdihedral_general

def synth_cnotdihedral_full(elem):
    if False:
        while True:
            i = 10
    'Decompose a CNOTDihedral element into a QuantumCircuit.\n    For N <= 2 qubits this is based on optimal CX cost decomposition from reference [1].\n    For N > 2 qubits this is done using the general non-optimal compilation routine from reference [2].\n\n    Args:\n        elem (CNOTDihedral): a CNOTDihedral element.\n    Return:\n        QuantumCircuit: a circuit implementation of the CNOTDihedral element.\n\n    References:\n        1. Shelly Garion and Andrew W. Cross, *Synthesis of CNOT-Dihedral circuits\n           with optimal number of two qubit gates*, `Quantum 4(369), 2020\n           <https://quantum-journal.org/papers/q-2020-12-07-369/>`_\n        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,\n           *Scalable randomised benchmarking of non-Clifford gates*,\n           npj Quantum Inf 2, 16012 (2016).\n    '
    num_qubits = elem.num_qubits
    if num_qubits < 3:
        return synth_cnotdihedral_two_qubits(elem)
    return synth_cnotdihedral_general(elem)