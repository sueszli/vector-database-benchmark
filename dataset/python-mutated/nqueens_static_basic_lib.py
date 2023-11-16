"""
    Simple, brute-force N-Queens solver. Using typing python
    Made by sebastiancr@fb.com(Sebastian Chaves) based on nqueens.py made by collinwinter@google.com (Collin Winter)
"""
from __future__ import annotations
import __static__
from typing import Generator, Tuple, Iterator

def permutations(iterable: Iterator[int], r: int=-1) -> Iterator[List[int]]:
    if False:
        while True:
            i = 10
    'permutations(range(3), 2) --> (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)'
    pool: List[int] = list(iterable)
    n: int = len(pool)
    if r == -1:
        r = n
    indices: List[int] = list(range(n))
    cycles: List[int] = list(range(n - r + 1, n + 1))[::-1]
    yield list((pool[i] for i in indices[:r]))
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                (indices[i], indices[-j]) = (indices[-j], indices[i])
                yield list((pool[i] for i in indices[:r]))
                break
        else:
            return

def n_queens(queen_count: int) -> Iterator[List[int]]:
    if False:
        print('Hello World!')
    'N-Queens solver.\n\n    Args:\n        queen_count: the number of queens to solve for. This is also the\n            board size.\n\n    Yields:\n        Solutions to the problem. Each yielded value is looks like\n        (3, 8, 2, 1, 4, ..., 6) where each number is the column position for the\n        queen, and the index into the tuple indicates the row.\n    '
    cols: Iterator[int] = range(queen_count)
    for vec in permutations(cols):
        if queen_count == len(set((vec[i] + i for i in cols))) == len(set((vec[i] - i for i in cols))):
            yield vec

def bench_n_queens(queen_count: int) -> List[List[int]]:
    if False:
        i = 10
        return i + 15
    return list(n_queens(queen_count))