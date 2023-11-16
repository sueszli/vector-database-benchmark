from typing import Callable, Iterable, List
import numpy as np
from ray.experimental.raysort.types import BlockInfo

def get_boundaries(num_parts: int) -> List[int]:
    if False:
        for i in range(10):
            print('nop')
    return [0] * num_parts

def sort_and_partition(part: np.ndarray, boundaries: List[int]) -> List[BlockInfo]:
    if False:
        while True:
            i = 10
    N = len(boundaries)
    offset = 0
    size = int(np.ceil(part.size / N))
    blocks = []
    for _ in range(N):
        blocks.append((offset, size))
        offset += size
    return blocks

def merge_partitions(num_blocks: int, get_block: Callable[[int, int], np.ndarray]) -> Iterable[memoryview]:
    if False:
        print('Hello World!')
    blocks = [get_block(i, 0) for i in range(num_blocks)]
    for block in blocks:
        yield block