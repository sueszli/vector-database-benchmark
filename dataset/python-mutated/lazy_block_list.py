import math
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats, _get_or_create_stats_actor
from ray.data._internal.util import _split_list
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata, BlockPartitionMetadata, MaybeBlockPartition
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef

class LazyBlockList(BlockList):
    """A BlockList that submits tasks lazily on-demand.

    This BlockList is used for implementing read operations (e.g., to avoid
    needing to read all files of a Dataset when the user is just wanting to
    .take() the first few rows or view the schema).
    """

    def __init__(self, tasks: List[ReadTask], read_stage_name: Optional[str]=None, block_partition_refs: Optional[List[ObjectRef[MaybeBlockPartition]]]=None, block_partition_meta_refs: Optional[List[ObjectRef[BlockMetadata]]]=None, cached_metadata: Optional[List[BlockPartitionMetadata]]=None, ray_remote_args: Optional[Dict[str, Any]]=None, stats_uuid: str=None, *, owned_by_consumer: bool):
        if False:
            i = 10
            return i + 15
        'Create a LazyBlockList on the provided read tasks.\n\n        Args:\n            tasks: The read tasks that will produce the blocks of this lazy block list.\n            read_stage_name: An optional name for the read stage, derived from the\n                underlying Datasource\n            block_partition_refs: An optional list of already submitted read task\n                futures (i.e. block partition refs). This should be the same length as\n                the tasks argument.\n            block_partition_meta_refs: An optional list of block partition metadata\n                refs. This should be the same length as the tasks argument.\n            cached_metadata: An optional list of already computed AND fetched metadata.\n                This serves as a cache of fetched block metadata. Note that each entry\n                in cached_metadata represents the list of output blocks metadata per\n                the read task. One task can produce multiple output blocks.\n            ray_remote_args: Ray remote arguments for the read tasks.\n            stats_uuid: UUID for the dataset stats, used to group and fetch read task\n                stats. If not provided, a new UUID will be created.\n        '
        self._tasks = tasks
        self._read_stage_name = read_stage_name
        self._num_blocks = len(self._tasks)
        if stats_uuid is None:
            stats_uuid = uuid.uuid4()
        self._stats_uuid = stats_uuid
        self._execution_started = False
        self._remote_args = ray_remote_args or {}
        if cached_metadata is not None:
            self._cached_metadata = cached_metadata
        else:
            self._cached_metadata = [None] * len(tasks)
        if block_partition_meta_refs is not None:
            self._block_partition_meta_refs = block_partition_meta_refs
        else:
            self._block_partition_meta_refs = [None] * len(tasks)
        if block_partition_refs is not None:
            self._block_partition_refs = block_partition_refs
        else:
            self._block_partition_refs = [None] * len(tasks)
        assert len(tasks) == len(self._block_partition_refs), (tasks, self._block_partition_refs)
        assert len(tasks) == len(self._block_partition_meta_refs), (tasks, self._block_partition_meta_refs)
        assert len(tasks) == len(self._cached_metadata), (tasks, self._cached_metadata)
        self._owned_by_consumer = owned_by_consumer
        self._stats_actor = _get_or_create_stats_actor()
        self._estimated_num_blocks = None

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'LazyBlockList(owned_by_consumer={self._owned_by_consumer})'

    def get_metadata(self, fetch_if_missing: bool=False) -> List[BlockMetadata]:
        if False:
            i = 10
            return i + 15
        'Get the metadata for all blocks.'
        if all((meta is not None for meta in self._cached_metadata)):
            metadata = self._flatten_metadata(self._cached_metadata)
        elif not fetch_if_missing:
            metadata = [m if m is not None else [t.get_metadata()] for (m, t) in zip(self._cached_metadata, self._tasks)]
            metadata = self._flatten_metadata(metadata)
        else:
            (_, metadata) = self._get_blocks_with_metadata()
        return metadata

    def stats(self) -> DatasetStats:
        if False:
            i = 10
            return i + 15
        'Create DatasetStats for this LazyBlockList.'
        return DatasetStats(stages={'Read': self.get_metadata(fetch_if_missing=False).copy()}, parent=None, needs_stats_actor=True, stats_uuid=self._stats_uuid)

    def copy(self) -> 'LazyBlockList':
        if False:
            return 10
        return LazyBlockList(self._tasks.copy(), read_stage_name=self._read_stage_name, block_partition_refs=self._block_partition_refs.copy(), block_partition_meta_refs=self._block_partition_meta_refs.copy(), cached_metadata=self._cached_metadata, ray_remote_args=self._remote_args.copy(), owned_by_consumer=self._owned_by_consumer, stats_uuid=self._stats_uuid)

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Clears all object references (block partitions and base block partitions)\n        from this lazy block list.\n        '
        self._block_partition_refs = [None for _ in self._block_partition_refs]
        self._block_partition_meta_refs = [None for _ in self._block_partition_meta_refs]
        self._cached_metadata = [None for _ in self._cached_metadata]
        self._stats_actor = None

    def is_cleared(self) -> bool:
        if False:
            return 10
        return all((ref is None for ref in self._block_partition_refs))

    def _check_if_cleared(self):
        if False:
            while True:
                i = 10
        pass

    def split(self, split_size: int) -> List['LazyBlockList']:
        if False:
            for i in range(10):
                print('nop')
        num_splits = math.ceil(len(self._tasks) / split_size)
        tasks = _split_list(self._tasks, num_splits)
        block_partition_refs = _split_list(self._block_partition_refs, num_splits)
        block_partition_meta_refs = _split_list(self._block_partition_meta_refs, num_splits)
        cached_metadata = _split_list(self._cached_metadata, num_splits)
        output = []
        for (t, b, m, c) in zip(tasks, block_partition_refs, block_partition_meta_refs, cached_metadata):
            output.append(LazyBlockList(t, b, m, c, owned_by_consumer=self._owned_by_consumer))
        return output

    def split_by_bytes(self, bytes_per_split: int) -> List['BlockList']:
        if False:
            while True:
                i = 10
        output = []
        (cur_tasks, cur_blocks, cur_blocks_meta, cur_cached_meta) = ([], [], [], [])
        cur_size = 0
        for (t, b, bm, c) in zip(self._tasks, self._block_partition_refs, self._block_partition_meta_refs, self._cached_metadata):
            m = t.get_metadata()
            if m.size_bytes is None:
                raise RuntimeError('Block has unknown size, cannot use split_by_bytes()')
            size = m.size_bytes
            if cur_blocks and cur_size + size > bytes_per_split:
                output.append(LazyBlockList(cur_tasks, cur_blocks, cur_blocks_meta, cur_cached_meta, owned_by_consumer=self._owned_by_consumer))
                (cur_tasks, cur_blocks, cur_blocks_meta, cur_cached_meta) = ([], [], [], [])
                cur_size = 0
            cur_tasks.append(t)
            cur_blocks.append(b)
            cur_blocks_meta.append(bm)
            cur_cached_meta.append(c)
            cur_size += size
        if cur_blocks:
            output.append(LazyBlockList(cur_tasks, cur_blocks, cur_blocks_meta, cur_cached_meta, owned_by_consumer=self._owned_by_consumer))
        return output

    def truncate_by_rows(self, limit: int) -> 'LazyBlockList':
        if False:
            while True:
                i = 10
        'Truncate the block list to the minimum number of blocks that contains at\n        least limit rows.\n\n        If the number of rows is not available, it will be treated as a 0-row block and\n        will be included in the truncated output.\n        '
        self._check_if_cleared()
        (out_tasks, out_blocks, out_blocks_meta, out_cached_meta) = ([], [], [], [])
        out_num_rows = 0
        for (t, b, bm, c) in zip(self._tasks, self._block_partition_refs, self._block_partition_meta_refs, self._cached_metadata):
            m = t.get_metadata()
            num_rows = m.num_rows
            if num_rows is None:
                num_rows = 0
            out_tasks.append(t)
            out_blocks.append(b)
            out_blocks_meta.append(bm)
            out_cached_meta.append(c)
            out_num_rows += num_rows
            if out_num_rows >= limit:
                break
        return LazyBlockList(out_tasks, out_blocks, out_blocks_meta, out_cached_meta, owned_by_consumer=self._owned_by_consumer)

    def divide(self, part_idx: int) -> ('LazyBlockList', 'LazyBlockList'):
        if False:
            for i in range(10):
                print('nop')
        left = LazyBlockList(self._tasks[:part_idx], self._block_partition_refs[:part_idx], self._block_partition_meta_refs[:part_idx], self._cached_metadata[:part_idx], owned_by_consumer=self._owned_by_consumer)
        right = LazyBlockList(self._tasks[part_idx:], self._block_partition_refs[part_idx:], self._block_partition_meta_refs[part_idx:], self._cached_metadata[part_idx:], owned_by_consumer=self._owned_by_consumer)
        return (left, right)

    def get_blocks(self) -> List[ObjectRef[Block]]:
        if False:
            for i in range(10):
                print('nop')
        "Bulk version of iter_blocks().\n\n        Prefer calling this instead of the iter form for performance if you\n        don't need lazy evaluation.\n        "
        (blocks, _) = self._get_blocks_with_metadata()
        return blocks

    def get_blocks_with_metadata(self) -> List[Tuple[ObjectRef[Block], BlockMetadata]]:
        if False:
            i = 10
            return i + 15
        "Bulk version of iter_blocks_with_metadata().\n\n        Prefer calling this instead of the iter form for performance if you\n        don't need lazy evaluation.\n        "
        (blocks, metadata) = self._get_blocks_with_metadata()
        return list(zip(blocks, metadata))

    def _get_blocks_with_metadata(self) -> Tuple[List[ObjectRef[Block]], List[BlockMetadata]]:
        if False:
            while True:
                i = 10
        'Get all underlying block futures and concrete metadata.\n\n        This will block on the completion of the underlying read tasks and will fetch\n        all block metadata outputted by those tasks.\n        '
        (block_refs, meta_refs) = ([], [])
        for (block_ref, meta_ref) in self._iter_block_partition_refs():
            block_refs.append(block_ref)
            meta_refs.append(meta_ref)
        read_progress_bar = ProgressBar('Read progress', total=len(block_refs))
        unique_refs = list(set(block_refs))
        generators = read_progress_bar.fetch_until_complete(unique_refs)
        ref_to_blocks = {}
        ref_to_metadata = {}
        for (ref, generator) in zip(unique_refs, generators):
            refs_list = list(generator)
            meta = ray.get(refs_list.pop(-1))
            ref_to_blocks[ref] = refs_list
            ref_to_metadata[ref] = meta
        output_block_refs = []
        for (idx, ref) in enumerate(block_refs):
            output_block_refs += ref_to_blocks[ref]
            self._cached_metadata[idx] = ref_to_metadata[ref]
        return (output_block_refs, self._flatten_metadata(self._cached_metadata))

    def compute_to_blocklist(self) -> BlockList:
        if False:
            for i in range(10):
                print('nop')
        'Launch all tasks and return a concrete BlockList.'
        (blocks, metadata) = self._get_blocks_with_metadata()
        return BlockList(blocks, metadata, owned_by_consumer=self._owned_by_consumer)

    def compute_first_block(self):
        if False:
            i = 10
            return i + 15
        'Kick off computation for the first block in the list.\n\n        This is useful if looking to support rapid lightweight interaction with a small\n        amount of the dataset.\n        '
        if self._tasks:
            self._get_or_compute(0)

    def ensure_metadata_for_first_block(self) -> Optional[BlockMetadata]:
        if False:
            return 10
        'Ensure that the metadata is fetched and set for the first block.\n\n        This will only block execution in order to fetch the post-read metadata for the\n        first block if the pre-read metadata for the first block has no schema.\n\n        Returns:\n            None if the block list is empty, the metadata for the first block otherwise.\n        '
        if not self._tasks:
            return None
        metadata = self._tasks[0].get_metadata()
        if metadata.schema is not None:
            return metadata
        try:
            (block_partition_ref, metadata_ref) = next(self._iter_block_partition_refs())
        except (StopIteration, ValueError):
            pass
        else:
            generator = ray.get(block_partition_ref)
            blocks_ref = list(generator)
            metadata = ray.get(blocks_ref[-1])
            self._cached_metadata[0] = metadata
        return metadata

    def iter_blocks(self) -> Iterator[ObjectRef[Block]]:
        if False:
            while True:
                i = 10
        'Iterate over the blocks of this block list.\n\n        This blocks on the execution of the tasks generating block outputs.\n        The length of this iterator is not known until execution.\n        '
        self._check_if_cleared()
        outer = self

        class Iter:

            def __init__(self):
                if False:
                    print('Hello World!')
                self._base_iter = outer.iter_blocks_with_metadata()

            def __iter__(self):
                if False:
                    print('Hello World!')
                return self

            def __next__(self):
                if False:
                    for i in range(10):
                        print('nop')
                (ref, meta) = next(self._base_iter)
                assert isinstance(ref, ray.ObjectRef), (ref, meta)
                return ref
        return Iter()

    def iter_blocks_with_metadata(self, block_for_metadata: bool=False) -> Iterator[Tuple[ObjectRef[Block], BlockMetadata]]:
        if False:
            i = 10
            return i + 15
        "Iterate over the blocks along with their metadata.\n\n        Note that, if block_for_metadata is False (default), this iterator returns\n        pre-read metadata from the ReadTasks given to this LazyBlockList so it doesn't\n        have to block on the execution of the read tasks. Therefore, the metadata may be\n        under-specified, e.g. missing schema or the number of rows. If fully-specified\n        block metadata is required, pass block_for_metadata=True. When dynamic block\n        splitting is enabled, always block on the execution of the read tasks.\n\n        The length of this iterator is not known until execution.\n\n        Args:\n            block_for_metadata: Whether we should block on the execution of read tasks\n                in order to obtain fully-specified block metadata.\n\n        Returns:\n            An iterator of block references and the corresponding block metadata.\n        "
        outer = self

        class Iter:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self._base_iter = outer._iter_block_partition_refs()
                self._pos = -1
                self._buffer = []

            def __iter__(self):
                if False:
                    print('Hello World!')
                return self

            def __next__(self):
                if False:
                    return 10
                while not self._buffer:
                    self._pos += 1
                    (generator_ref, _) = next(self._base_iter)
                    generator = ray.get(generator_ref)
                    refs = list(generator)
                    metadata = ray.get(refs.pop(-1))
                    assert len(metadata) == len(refs)
                    for (block_ref, meta) in zip(refs, metadata):
                        self._buffer.append((block_ref, meta))
                return self._buffer.pop(0)
        return Iter()

    def randomize_block_order(self, seed: Optional[int]=None) -> 'LazyBlockList':
        if False:
            return 10
        'Randomizes the order of the blocks.\n\n        Args:\n            seed: Fix the random seed to use, otherwise one will be chosen\n                based on system randomness.\n        '
        import random
        if seed is not None:
            random.seed(seed)
        zipped = list(zip(self._tasks, self._block_partition_refs, self._block_partition_meta_refs, self._cached_metadata))
        random.shuffle(zipped)
        (tasks, block_partition_refs, block_partition_meta_refs, cached_metadata) = map(list, zip(*zipped))
        return LazyBlockList(tasks, block_partition_refs=block_partition_refs, block_partition_meta_refs=block_partition_meta_refs, cached_metadata=cached_metadata, ray_remote_args=self._remote_args.copy(), owned_by_consumer=self._owned_by_consumer, stats_uuid=self._stats_uuid)

    def _iter_block_partition_refs(self) -> Iterator[Tuple[ObjectRef[MaybeBlockPartition], Union[None, ObjectRef[BlockMetadata]]]]:
        if False:
            return 10
        'Iterate over the block futures and their corresponding metadata futures.\n\n        This does NOT block on the execution of each submitted task.\n        '
        outer = self

        class Iter:

            def __init__(self):
                if False:
                    print('Hello World!')
                self._pos = -1

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                return self

            def __next__(self):
                if False:
                    return 10
                self._pos += 1
                if self._pos < len(outer._tasks):
                    return outer._get_or_compute(self._pos)
                raise StopIteration
        return Iter()

    def _get_or_compute(self, i: int) -> Tuple[ObjectRef[MaybeBlockPartition], Union[None, ObjectRef[BlockMetadata]]]:
        if False:
            for i in range(10):
                print('nop')
        assert i < len(self._tasks), i
        if not self._block_partition_refs[i]:
            for j in range(max(i + 1, i * 2)):
                if j >= len(self._block_partition_refs):
                    break
                if not self._block_partition_refs[j]:
                    (self._block_partition_refs[j], self._block_partition_meta_refs[j]) = self._submit_task(j)
            assert self._block_partition_refs[i], self._block_partition_refs
        trace_allocation(self._block_partition_refs[i], f'LazyBlockList.get_or_compute({i})')
        return (self._block_partition_refs[i], self._block_partition_meta_refs[i])

    def _submit_task(self, task_idx: int) -> Tuple[ObjectRef[MaybeBlockPartition], Union[None, ObjectRef[BlockMetadata]]]:
        if False:
            print('Hello World!')
        'Submit the task with index task_idx.\n\n        NOTE: When dynamic block splitting is enabled, returns\n        Tuple[ObjectRef[ObjectRefGenerator], None] instead of\n        Tuple[ObjectRef[Block], ObjectRef[BlockMetadata]], and the blocks metadata will\n        be fetched as the last element in ObjectRefGenerator.\n        '
        if self._stats_actor is None:
            self._stats_actor = _get_or_create_stats_actor()
        stats_actor = self._stats_actor
        if not self._execution_started:
            ray.get(stats_actor.record_start.remote(self._stats_uuid))
            self._execution_started = True
        task = self._tasks[task_idx]
        return (cached_remote_fn(_execute_read_task_split).options(num_returns='dynamic', **self._remote_args).remote(i=task_idx, task=task, context=DataContext.get_current(), stats_uuid=self._stats_uuid, stats_actor=stats_actor), None)

    def _num_computed(self) -> int:
        if False:
            print('Hello World!')
        i = 0
        for b in self._block_partition_refs:
            if b is not None:
                i += 1
        return i

    def _flatten_metadata(self, metadata: List[BlockPartitionMetadata]) -> List[BlockMetadata]:
        if False:
            return 10
        'Flatten the metadata of computed blocks into a list.\n\n        This is required because dynamic block splitting can produce multiple output\n        blocks from each task.\n        '
        return [meta for meta_list in metadata for meta in meta_list]

def _execute_read_task_nosplit(i: int, task: ReadTask, context: DataContext, stats_uuid: str, stats_actor: ray.actor.ActorHandle) -> Tuple[Block, BlockMetadata]:
    if False:
        i = 10
        return i + 15
    DataContext._set_current(context)
    stats = BlockExecStats.builder()
    blocks = list(task())
    assert len(blocks) == 1
    block = blocks[0]
    metadata = task.get_metadata()
    metadata = BlockAccessor.for_block(block).get_metadata(input_files=metadata.input_files, exec_stats=stats.build())
    stats_actor.record_task.remote(stats_uuid, i, [metadata])
    return (block, metadata)

def _execute_read_task_split(i: int, task: ReadTask, context: DataContext, stats_uuid: str, stats_actor: ray.actor.ActorHandle) -> Iterable[Union[Block, List[BlockMetadata]]]:
    if False:
        print('Hello World!')
    'Execute read task with dynamic block splitting.\n\n    Returns an Iterable of blocks followed by their metadata.\n    Example of return value for 3 blocks:\n    (Block1, Block2, Block3, [BlockMetadata1, BlockMetadata2, BlockMetadata3])\n    '
    DataContext._set_current(context)
    blocks = task()
    input_files = task.get_metadata().input_files
    blocks_metadata = []
    block_exec_stats = BlockExecStats.builder()
    for block in blocks:
        metadata = BlockAccessor.for_block(block).get_metadata(input_files=input_files, exec_stats=block_exec_stats.build())
        yield block
        blocks_metadata.append(metadata)
        block_exec_stats = BlockExecStats.builder()
    stats_actor.record_task.remote(stats_uuid, i, blocks_metadata)
    yield blocks_metadata