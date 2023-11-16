"""
Circuit synthesis for the Clifford class for all-to-all architecture.
"""
from qiskit.synthesis.clifford.clifford_decompose_ag import synth_clifford_ag
from qiskit.synthesis.clifford.clifford_decompose_bm import synth_clifford_bm
from qiskit.synthesis.clifford.clifford_decompose_greedy import synth_clifford_greedy

def synth_clifford_full(clifford, method=None):
    if False:
        for i in range(10):
            print('nop')
    "Decompose a Clifford operator into a QuantumCircuit.\n\n    For N <= 3 qubits this is based on optimal CX cost decomposition\n    from reference [1]. For N > 3 qubits this is done using the general\n    non-optimal greedy compilation routine from reference [3],\n    which typically yields better CX cost compared to the AG method in [2].\n\n    Args:\n        clifford (Clifford): a clifford operator.\n        method (str):  Optional, a synthesis method ('AG' or 'greedy').\n             If set this overrides optimal decomposition for N <=3 qubits.\n\n    Return:\n        QuantumCircuit: a circuit implementation of the Clifford.\n\n    References:\n        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the\n           structure of the Clifford group*,\n           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_\n\n        2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,\n           Phys. Rev. A 70, 052328 (2004).\n           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_\n\n        3. Sergey Bravyi, Shaohan Hu, Dmitri Maslov, Ruslan Shaydulin,\n           *Clifford Circuit Optimization with Templates and Symbolic Pauli Gates*,\n           `arXiv:2105.02291 [quant-ph] <https://arxiv.org/abs/2105.02291>`_\n    "
    num_qubits = clifford.num_qubits
    if method == 'AG':
        return synth_clifford_ag(clifford)
    if method == 'greedy':
        return synth_clifford_greedy(clifford)
    if num_qubits <= 3:
        return synth_clifford_bm(clifford)
    return synth_clifford_greedy(clifford)