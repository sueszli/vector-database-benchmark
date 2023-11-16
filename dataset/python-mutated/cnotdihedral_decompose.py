"""
Circuit synthesis for the CNOTDihedral class.
"""
from __future__ import annotations
from qiskit.synthesis.cnotdihedral import synth_cnotdihedral_two_qubits, synth_cnotdihedral_general
from qiskit.utils.deprecation import deprecate_func

@deprecate_func(additional_msg='Instead, use the function qiskit.synthesis.synth_cnotdihedral_full.', since='0.23.0', package_name='qiskit-terra')
def decompose_cnotdihedral(elem):
    if False:
        return 10
    'DEPRECATED: Decompose a CNOTDihedral element into a QuantumCircuit.\n\n    Args:\n        elem (CNOTDihedral): a CNOTDihedral element.\n    Return:\n        QuantumCircuit: a circuit implementation of the CNOTDihedral element.\n\n    References:\n        1. Shelly Garion and Andrew W. Cross, *Synthesis of CNOT-Dihedral circuits\n           with optimal number of two qubit gates*, `Quantum 4(369), 2020\n           <https://quantum-journal.org/papers/q-2020-12-07-369/>`_\n        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,\n           *Scalable randomised benchmarking of non-Clifford gates*,\n           npj Quantum Inf 2, 16012 (2016).\n    '
    num_qubits = elem.num_qubits
    if num_qubits < 3:
        return synth_cnotdihedral_two_qubits(elem)
    return synth_cnotdihedral_general(elem)