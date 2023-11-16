import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.fast_repartition import fast_repartition
from ray.data._internal.plan import AllToAllStage
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.shuffle_and_partition import PushBasedShufflePartitionOp, SimpleShufflePartitionOp
from ray.data._internal.sort import SortKey, sort_impl
from ray.data._internal.split import _split_at_index, _split_at_indices
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata, BlockPartition
from ray.data.context import DataContext
if TYPE_CHECKING:
    from ray.data import Dataset

class RepartitionStage(AllToAllStage):
    """Implementation of `Dataset.repartition()`."""

    def __init__(self, num_blocks: int, shuffle: bool):
        if False:
            for i in range(10):
                print('nop')
        if shuffle:

            def do_shuffle(block_list, ctx: TaskContext, clear_input_blocks: bool, block_udf, remote_args):
                if False:
                    return 10
                if clear_input_blocks:
                    blocks = block_list.copy()
                    block_list.clear()
                else:
                    blocks = block_list
                context = DataContext.get_current()
                if context.use_push_based_shuffle:
                    shuffle_op_cls = PushBasedShufflePartitionOp
                else:
                    shuffle_op_cls = SimpleShufflePartitionOp
                shuffle_op = shuffle_op_cls(block_udf, random_shuffle=False)
                return shuffle_op.execute(blocks, num_blocks, clear_input_blocks, map_ray_remote_args=remote_args, reduce_ray_remote_args=remote_args, ctx=ctx)
            super().__init__('Repartition', num_blocks, do_shuffle, supports_block_udf=True, sub_stage_names=['ShuffleMap', 'ShuffleReduce'])
        else:

            def do_fast_repartition(block_list, ctx: TaskContext, clear_input_blocks: bool, *_):
                if False:
                    print('Hello World!')
                if clear_input_blocks:
                    blocks = block_list.copy()
                    block_list.clear()
                else:
                    blocks = block_list
                return fast_repartition(blocks, num_blocks, ctx)
            super().__init__('Repartition', num_blocks, do_fast_repartition, sub_stage_names=['Repartition'])

class RandomizeBlocksStage(AllToAllStage):
    """Implementation of `Dataset.randomize_blocks()`."""

    def __init__(self, seed: Optional[int]):
        if False:
            return 10
        self._seed = seed
        super().__init__('RandomizeBlockOrder', None, self.do_randomize)

    def do_randomize(self, block_list, *_):
        if False:
            for i in range(10):
                print('nop')
        num_blocks = block_list.initial_num_blocks()
        if num_blocks == 0:
            return (block_list, {})
        randomized_block_list = block_list.randomize_block_order(self._seed)
        return (randomized_block_list, {})

class RandomShuffleStage(AllToAllStage):
    """Implementation of `Dataset.random_shuffle()`."""

    def __init__(self, seed: Optional[int], output_num_blocks: Optional[int], remote_args: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')

        def do_shuffle(block_list, ctx: TaskContext, clear_input_blocks: bool, block_udf, remote_args):
            if False:
                print('Hello World!')
            num_blocks = block_list.executed_num_blocks()
            if num_blocks == 0:
                return (block_list, {})
            if clear_input_blocks:
                blocks = block_list.copy()
                block_list.clear()
            else:
                blocks = block_list
            context = DataContext.get_current()
            if context.use_push_based_shuffle:
                if output_num_blocks is not None:
                    raise NotImplementedError("Push-based shuffle doesn't support setting num_blocks yet.")
                shuffle_op_cls = PushBasedShufflePartitionOp
            else:
                shuffle_op_cls = SimpleShufflePartitionOp
            random_shuffle_op = shuffle_op_cls(block_udf, random_shuffle=True, random_seed=seed)
            return random_shuffle_op.execute(blocks, output_num_blocks or num_blocks, clear_input_blocks, map_ray_remote_args=remote_args, reduce_ray_remote_args=remote_args, ctx=ctx)
        super().__init__('RandomShuffle', output_num_blocks, do_shuffle, supports_block_udf=True, remote_args=remote_args, sub_stage_names=['ShuffleMap', 'ShuffleReduce'])

class ZipStage(AllToAllStage):
    """Implementation of `Dataset.zip()`."""

    def __init__(self, other: 'Dataset'):
        if False:
            return 10

        def do_zip_all(block_list: BlockList, clear_input_blocks: bool, *_):
            if False:
                return 10
            base_block_list = block_list
            base_blocks_with_metadata = block_list.get_blocks_with_metadata()
            (base_block_rows, base_block_bytes) = _calculate_blocks_rows_and_bytes(base_blocks_with_metadata)
            other_block_list = other._plan.execute(preserve_order=True)
            other_blocks_with_metadata = other_block_list.get_blocks_with_metadata()
            (other_block_rows, other_block_bytes) = _calculate_blocks_rows_and_bytes(other_blocks_with_metadata)
            inverted = False
            if sum(other_block_bytes) > sum(base_block_bytes):
                (base_block_list, other_block_list) = (other_block_list, base_block_list)
                (base_blocks_with_metadata, other_blocks_with_metadata) = (other_blocks_with_metadata, base_blocks_with_metadata)
                (base_block_rows, other_block_rows) = (other_block_rows, base_block_rows)
                inverted = True
            indices = list(itertools.accumulate(base_block_rows))
            indices.pop(-1)
            total_base_rows = sum(base_block_rows)
            total_other_rows = sum(other_block_rows)
            if total_base_rows != total_other_rows:
                raise ValueError(f'Cannot zip datasets of different number of rows: {total_base_rows}, {total_other_rows}')
            aligned_other_blocks_with_metadata = _split_at_indices(other_blocks_with_metadata, indices, other_block_list._owned_by_consumer, other_block_rows)
            del other_blocks_with_metadata
            base_blocks = [b for (b, _) in base_blocks_with_metadata]
            other_blocks = aligned_other_blocks_with_metadata[0]
            del base_blocks_with_metadata, aligned_other_blocks_with_metadata
            if clear_input_blocks:
                base_block_list.clear()
                other_block_list.clear()
            do_zip = cached_remote_fn(_do_zip, num_returns=2)
            out_blocks = []
            out_metadata = []
            for (base_block, other_blocks) in zip(base_blocks, other_blocks):
                (res, meta) = do_zip.remote(base_block, *other_blocks, inverted=inverted)
                out_blocks.append(res)
                out_metadata.append(meta)
            del base_blocks, other_blocks
            out_metadata = ray.get(out_metadata)
            blocks = BlockList(out_blocks, out_metadata, owned_by_consumer=base_block_list._owned_by_consumer)
            return (blocks, {})
        super().__init__('Zip', None, do_zip_all)

def _calculate_blocks_rows_and_bytes(blocks_with_metadata: BlockPartition) -> Tuple[List[int], List[int]]:
    if False:
        return 10
    'Calculate the number of rows and size in bytes for a list of blocks with\n    metadata.\n    '
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

def _get_num_rows_and_bytes(block: Block) -> Tuple[int, int]:
    if False:
        while True:
            i = 10
    block = BlockAccessor.for_block(block)
    return (block.num_rows(), block.size_bytes())

def _do_zip(block: Block, *other_blocks: Block, inverted: bool=False) -> Tuple[Block, BlockMetadata]:
    if False:
        i = 10
        return i + 15
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

class SortStage(AllToAllStage):
    """Implementation of `Dataset.sort()`."""

    def __init__(self, ds: 'Dataset', sort_key: SortKey):
        if False:
            i = 10
            return i + 15

        def do_sort(block_list, ctx: TaskContext, clear_input_blocks: bool, *_):
            if False:
                return 10
            if block_list.initial_num_blocks() == 0:
                return (block_list, {})
            if clear_input_blocks:
                blocks = block_list.copy()
                block_list.clear()
            else:
                blocks = block_list
            sort_key.validate_schema(ds.schema(fetch_if_missing=True))
            return sort_impl(blocks, clear_input_blocks, sort_key, ctx)
        super().__init__('Sort', None, do_sort, sub_stage_names=['SortSample', 'ShuffleMap', 'ShuffleReduce'])

class LimitStage(AllToAllStage):
    """Implementation of `Dataset.limit()`."""

    def __init__(self, limit: int):
        if False:
            print('Hello World!')
        self._limit = limit
        super().__init__('Limit', None, self._do_limit)

    @property
    def limit(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._limit

    def _do_limit(self, input_block_list: BlockList, clear_input_blocks: bool, *_):
        if False:
            i = 10
            return i + 15
        if clear_input_blocks:
            block_list = input_block_list.copy()
            input_block_list.clear()
        else:
            block_list = input_block_list
        block_list = block_list.truncate_by_rows(self._limit)
        (blocks, metadata, _, _) = _split_at_index(block_list, self._limit)
        return (BlockList(blocks, metadata, owned_by_consumer=block_list._owned_by_consumer), {})