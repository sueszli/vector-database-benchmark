"""Synthesis algorithm for Permutation gates for full-connectivity."""
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .permutation_utils import _get_ordered_swap, _inverse_pattern, _pattern_to_cycles, _decompose_cycles

def synth_permutation_basic(pattern):
    if False:
        i = 10
        return i + 15
    'Synthesize a permutation circuit for a fully-connected\n    architecture using sorting.\n\n    More precisely, if the input permutation is a cycle of length ``m``,\n    then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);\n    if the input  permutation consists of several disjoint cycles, then each cycle\n    is essentially treated independently.\n\n    Args:\n        pattern (Union[list[int], np.ndarray]): permutation pattern, describing\n            which qubits occupy the positions 0, 1, 2, etc. after applying the\n            permutation. That is, ``pattern[k] = m`` when the permutation maps\n            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``\n            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to\n            position ``1``, etc.\n\n    Returns:\n        QuantumCircuit: the synthesized quantum circuit.\n    '
    num_qubits = len(pattern)
    qc = QuantumCircuit(num_qubits)
    swaps = _get_ordered_swap(pattern)
    for swap in swaps:
        qc.swap(swap[0], swap[1])
    return qc

def synth_permutation_acg(pattern):
    if False:
        while True:
            i = 10
    'Synthesize a permutation circuit for a fully-connected\n    architecture using the Alon, Chung, Graham method.\n\n    This produces a quantum circuit of depth 2 (measured in the number of SWAPs).\n\n    This implementation is based on the Theorem 2 in the paper\n    "Routing Permutations on Graphs Via Matchings" (1993),\n    available at https://www.cs.tau.ac.il/~nogaa/PDFS/r.pdf.\n\n    Args:\n        pattern (Union[list[int], np.ndarray]): permutation pattern, describing\n            which qubits occupy the positions 0, 1, 2, etc. after applying the\n            permutation. That is, ``pattern[k] = m`` when the permutation maps\n            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``\n            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to\n            position ``1``, etc.\n\n    Returns:\n        QuantumCircuit: the synthesized quantum circuit.\n    '
    num_qubits = len(pattern)
    qc = QuantumCircuit(num_qubits)
    cur_pattern = _inverse_pattern(pattern)
    cycles = _pattern_to_cycles(cur_pattern)
    swaps = _decompose_cycles(cycles)
    for swap in swaps:
        qc.swap(swap[0], swap[1])
    return qc