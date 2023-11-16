"""Implements "Block Partitions of Sequences" by Imre Bárány et al.

Paper: https://arxiv.org/pdf/1308.2452.pdf

"""
from typing import Iterator, List, Tuple
__all__ = ['solve']

def solve(sequence: List[int], partitions: int=1) -> List[List[int]]:
    if False:
        print('Hello World!')
    'Splits a sequence into several partitions to minimize variance for each\n    partition.\n\n    The result might not be optimal. However, it can be done only in O(kn³),\n    where k is the number of partitions and n is the length of the sequence.\n\n    '
    if partitions < 1:
        raise ValueError(f'partitions must be a positive integer ({partitions} < 1)')
    n = len(sequence)
    if n < partitions:
        raise ValueError(f'sequence is shorter than intended partitions ({n} < {partitions})')
    minimum = min(sequence)
    maximum = max(sequence) - minimum
    normal_sequence: List[float]
    if maximum == 0:
        normal_sequence = [0 for _ in sequence]
    else:
        normal_sequence = [(x - minimum) / maximum for x in sequence]
    splits = [n // partitions * (x + 1) for x in range(partitions - 1)] + [n]

    def block_size(i: int) -> float:
        if False:
            print('Hello World!')
        start = splits[i - 1] if i > 0 else 0
        stop = splits[i]
        return sum(normal_sequence[start:stop])

    def leaderboard() -> Iterator[Tuple[float, int]]:
        if False:
            print('Hello World!')
        return ((block_size(i), i) for i in range(partitions))
    while True:
        '\n        (1) Fix p ∈ [k] with M(P) = bp. So Bp is a maximal block of P.\n        '
        (max_size, p) = max(leaderboard())
        while True:
            '\n            (2) If M(P) ≤ m(P) + 1, then stop.\n            '
            (min_size, q) = min(leaderboard())
            if max_size <= min_size + 1:
                return [sequence[i:j] for (i, j) in zip([0] + splits[:-1], splits)]
            '\n            (3) If M(P) > m(P) + 1, then let m(P) = bq for the q ∈ [k] which is\n            closest to p (ties broken arbitrarily). Thus Bq is a minimal block\n            of P. Let Bh be the block next to Bq between Bp and Bq. (Note that\n            Bh is a non-empty block: if it were, then m(P) = 0 and we should\n            have chosen Bh instead of Bq.)\n            '
            if p < q:
                '\n                So either p < q and then h = q−1 and we define P ∗ by moving\n                the last element from Bh = Bq−1 to Bq,\n                '
                h = q - 1
                splits[h] -= 1
            else:
                '\n                or q < p, and then h = q + 1 and P ∗ is obtained by moving the\n                first element of Bh = Bq+1 to Bq.\n                '
                h = q + 1
                splits[q] += 1
            '\n            Set P = P ∗ . If p = h, then go to (1), else go to (2).\n            '
            if p == h:
                break