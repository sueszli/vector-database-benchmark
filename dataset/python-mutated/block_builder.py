from typing import Generic
from ray.data.block import Block, BlockAccessor, T

class BlockBuilder(Generic[T]):
    """A builder class for blocks."""

    @staticmethod
    def for_block(block: Block) -> 'BlockBuilder':
        if False:
            i = 10
            return i + 15
        return BlockAccessor.for_block(block).builder()

    def add(self, item: T) -> None:
        if False:
            while True:
                i = 10
        'Append a single row to the block being built.'
        raise NotImplementedError

    def add_block(self, block: Block) -> None:
        if False:
            while True:
                i = 10
        'Append an entire block to the block being built.'
        raise NotImplementedError

    def will_build_yield_copy(self) -> bool:
        if False:
            print('Hello World!')
        'Whether building this block will yield a new block copy.'
        raise NotImplementedError

    def build(self) -> Block:
        if False:
            while True:
                i = 10
        'Build the block.'
        raise NotImplementedError

    def num_rows(self) -> int:
        if False:
            print('Hello World!')
        'Return the number of rows added in the block.'
        raise NotImplementedError

    def get_estimated_memory_usage(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the estimated memory usage so far in bytes.'
        raise NotImplementedError