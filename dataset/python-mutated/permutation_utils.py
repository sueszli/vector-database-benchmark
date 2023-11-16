"""Utility functions for handling permutations."""

def _get_ordered_swap(permutation_in):
    if False:
        print('Hello World!')
    "Sorts the input permutation by iterating through the permutation list\n    and putting each element to its correct position via a SWAP (if it's not\n    at the correct position already). If ``n`` is the length of the input\n    permutation, this requires at most ``n`` SWAPs.\n\n    More precisely, if the input permutation is a cycle of length ``m``,\n    then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);\n    if the input  permutation consists of several disjoint cycles, then each cycle\n    is essentially treated independently.\n    "
    permutation = list(permutation_in[:])
    swap_list = []
    index_map = _inverse_pattern(permutation_in)
    for (i, val) in enumerate(permutation):
        if val != i:
            j = index_map[i]
            swap_list.append((i, j))
            (permutation[i], permutation[j]) = (permutation[j], permutation[i])
            index_map[val] = j
            index_map[i] = i
    swap_list.reverse()
    return swap_list

def _inverse_pattern(pattern):
    if False:
        print('Hello World!')
    'Finds inverse of a permutation pattern.'
    b_map = {pos: idx for (idx, pos) in enumerate(pattern)}
    return [b_map[pos] for pos in range(len(pattern))]

def _pattern_to_cycles(pattern):
    if False:
        return 10
    'Given a permutation pattern, creates its disjoint cycle decomposition.'
    nq = len(pattern)
    explored = [False] * nq
    cycles = []
    for i in pattern:
        cycle = []
        while not explored[i]:
            cycle.append(i)
            explored[i] = True
            i = pattern[i]
        if len(cycle) >= 2:
            cycles.append(cycle)
    return cycles

def _decompose_cycles(cycles):
    if False:
        for i in range(10):
            print('nop')
    'Given a disjoint cycle decomposition, decomposes every cycle into a SWAP\n    circuit of depth 2.'
    swap_list = []
    for cycle in cycles:
        m = len(cycle)
        for i in range((m - 1) // 2):
            swap_list.append((cycle[i - 1], cycle[m - 3 - i]))
        for i in range(m // 2):
            swap_list.append((cycle[i - 1], cycle[m - 2 - i]))
    return swap_list