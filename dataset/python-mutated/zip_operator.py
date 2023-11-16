import itertools
from typing import List, Tuple
import ray
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.split import _split_at_indices
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata, BlockPartition

class ZipOperator(PhysicalOperator):
    """An operator that zips its inputs together.

    NOTE: the implementation is bulk for now, which materializes all its inputs in
    object store, before starting execution. Should re-implement it as a streaming
    operator in the future.
    """

    def __init__(self, left_input_op: PhysicalOperator, right_input_op: PhysicalOperator):
        if False:
            return 10
        'Create a ZipOperator.\n\n        Args:\n            left_input_ops: The input operator at left hand side.\n            right_input_op: The input operator at right hand side.\n        '
        self._left_buffer: List[RefBundle] = []
        self._right_buffer: List[RefBundle] = []
        self._output_buffer: List[RefBundle] = []
        self._stats: StatsDict = {}
        super().__init__('Zip', [left_input_op, right_input_op], target_max_block_size=None)

    def num_outputs_total(self) -> int:
        if False:
            return 10
        left_num_outputs = self.input_dependencies[0].num_outputs_total()
        right_num_outputs = self.input_dependencies[1].num_outputs_total()
        if left_num_outputs is not None and right_num_outputs is not None:
            return max(left_num_outputs, right_num_outputs)
        elif left_num_outputs is not None:
            return left_num_outputs
        else:
            return right_num_outputs

    def _add_input_inner(self, refs: RefBundle, input_index: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert not self.completed()
        assert input_index == 0 or input_index == 1, input_index
        if input_index == 0:
            self._left_buffer.append(refs)
        else:
            self._right_buffer.append(refs)

    def all_inputs_done(self) -> None:
        if False:
            while True:
                i = 10
        (self._output_buffer, self._stats) = self._zip(self._left_buffer, self._right_buffer)
        self._left_buffer.clear()
        self._right_buffer.clear()
        super().all_inputs_done()

    def has_next(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(self._output_buffer) > 0

    def _get_next_inner(self) -> RefBundle:
        if False:
            print('Hello World!')
        return self._output_buffer.pop(0)

    def get_stats(self) -> StatsDict:
        if False:
            return 10
        return self._stats

    def _zip(self, left_input: List[RefBundle], right_input: List[RefBundle]) -> Tuple[List[RefBundle], StatsDict]:
        if False:
            while True:
                i = 10
        'Zip the RefBundles from `left_input` and `right_input` together.\n\n        Zip is done in 2 steps: aligning blocks, and zipping blocks from\n        both sides.\n\n        Aligning blocks (optional): check the blocks from `left_input` and\n        `right_input` are aligned or not, i.e. if having different number of blocks, or\n        having different number of rows in some blocks. If not aligned, repartition the\n        smaller input with `_split_at_indices` to align with larger input.\n\n        Zipping blocks: after blocks from both sides are aligned, zip\n        blocks from both sides together in parallel.\n        '
        left_blocks_with_metadata = []
        for bundle in left_input:
            for (block, meta) in bundle.blocks:
                left_blocks_with_metadata.append((block, meta))
        right_blocks_with_metadata = []
        for bundle in right_input:
            for (block, meta) in bundle.blocks:
                right_blocks_with_metadata.append((block, meta))
        (left_block_rows, left_block_bytes) = self._calculate_blocks_rows_and_bytes(left_blocks_with_metadata)
        (right_block_rows, right_block_bytes) = self._calculate_blocks_rows_and_bytes(right_blocks_with_metadata)
        total_left_rows = sum(left_block_rows)
        total_right_rows = sum(right_block_rows)
        if total_left_rows != total_right_rows:
            raise ValueError(f'Cannot zip datasets of different number of rows: {total_left_rows}, {total_right_rows}')
        input_side_inverted = False
        if sum(right_block_bytes) > sum(left_block_bytes):
            (left_blocks_with_metadata, right_blocks_with_metadata) = (right_blocks_with_metadata, left_blocks_with_metadata)
            (left_block_rows, right_block_rows) = (right_block_rows, left_block_rows)
            input_side_inverted = True
        indices = list(itertools.accumulate(left_block_rows))
        indices.pop(-1)
        aligned_right_blocks_with_metadata = _split_at_indices(right_blocks_with_metadata, indices, block_rows=right_block_rows)
        del right_blocks_with_metadata
        left_blocks = [b for (b, _) in left_blocks_with_metadata]
        right_blocks_list = aligned_right_blocks_with_metadata[0]
        del left_blocks_with_metadata, aligned_right_blocks_with_metadata
        zip_one_block = cached_remote_fn(_zip_one_block, num_returns=2)
        output_blocks = []
        output_metadata = []
        for (left_block, right_blocks) in zip(left_blocks, right_blocks_list):
            (res, meta) = zip_one_block.remote(left_block, *right_blocks, inverted=input_side_inverted)
            output_blocks.append(res)
            output_metadata.append(meta)
        del left_blocks, right_blocks_list
        output_metadata = ray.get(output_metadata)
        output_refs = []
        input_owned = all((b.owns_blocks for b in left_input))
        for (block, meta) in zip(output_blocks, output_metadata):
            output_refs.append(RefBundle([(block, meta)], owns_blocks=input_owned))
        stats = {self._name: output_metadata}
        for ref in left_input:
            ref.destroy_if_owned()
        for ref in right_input:
            ref.destroy_if_owned()
        return (output_refs, stats)

    def _calculate_blocks_rows_and_bytes(self, blocks_with_metadata: BlockPartition) -> Tuple[List[int], List[int]]:
        if False:
            while True:
                i = 10
        'Calculate the number of rows and size in bytes for a list of blocks with\n        metadata.\n        '
        get_num_rows_and_bytes = cached_remote_fn(_get_num_rows_and_bytes)
        block_rows = []
        block_bytes = []
        for (block, metadata) in blocks_with_metadata:
            if metadata.num_rows is None or metadata.size_bytes is None:
                (num_rows, size_bytes) = ray.get(get_num_rows_and_bytes.remote(block))
                metadata.num_rows = num_rows
                metadata.size_bytes = size_bytes
            block_rows.append(metadata.num_rows)
            block_bytes.append(metadata.size_bytes)
        return (block_rows, block_bytes)

def _zip_one_block(block: Block, *other_blocks: Block, inverted: bool=False) -> Tuple[Block, BlockMetadata]:
    if False:
        for i in range(10):
            print('nop')
    'Zip together `block` with `other_blocks`.'
    stats = BlockExecStats.builder()
    builder = DelegatingBlockBuilder()
    for other_block in other_blocks:
        builder.add_block(other_block)
    other_block = builder.build()
    if inverted:
        (block, other_block) = (other_block, block)
    result = BlockAccessor.for_block(block).zip(other_block)
    br = BlockAccessor.for_block(result)
    return (result, br.get_metadata(input_files=[], exec_stats=stats.build()))

def _get_num_rows_and_bytes(block: Block) -> Tuple[int, int]:
    if False:
        for i in range(10):
            print('nop')
    block = BlockAccessor.for_block(block)
    return (block.num_rows(), block.size_bytes())