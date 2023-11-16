"""Depth-efficient synthesis algorithm for Permutation gates."""
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .permutation_utils import _inverse_pattern

def synth_permutation_depth_lnn_kms(pattern):
    if False:
        print('Hello World!')
    'Synthesize a permutation circuit for a linear nearest-neighbor\n    architecture using the Kutin, Moulton, Smithline method.\n\n    This is the permutation synthesis algorithm from\n    https://arxiv.org/abs/quant-ph/0701194, Chapter 6.\n    It synthesizes any permutation of n qubits over linear nearest-neighbor\n    architecture using SWAP gates with depth at most n and size at most\n    n(n-1)/2 (where both depth and size are measured with respect to SWAPs).\n\n    Args:\n        pattern (Union[list[int], np.ndarray]): permutation pattern, describing\n            which qubits occupy the positions 0, 1, 2, etc. after applying the\n            permutation. That is, ``pattern[k] = m`` when the permutation maps\n            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``\n            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to\n            position ``1``, etc.\n\n    Returns:\n        QuantumCircuit: the synthesized quantum circuit.\n    '
    cur_pattern = _inverse_pattern(pattern)
    num_qubits = len(cur_pattern)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        _create_swap_layer(qc, cur_pattern, i % 2)
    return qc

def _create_swap_layer(qc, pattern, starting_point):
    if False:
        while True:
            i = 10
    'Implements a single swap layer, consisting of conditional swaps between each\n    neighboring couple. The starting_point is the first qubit to use (either 0 or 1\n    for even or odd layers respectively). Mutates both the quantum circuit ``qc``\n    and the permutation pattern ``pattern``.\n    '
    num_qubits = len(pattern)
    for j in range(starting_point, num_qubits - 1, 2):
        if pattern[j] > pattern[j + 1]:
            qc.swap(j, j + 1)
            (pattern[j], pattern[j + 1]) = (pattern[j + 1], pattern[j])