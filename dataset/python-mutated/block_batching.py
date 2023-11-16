import collections
import itertools
from contextlib import nullcontext
from typing import Any, Callable, Iterator, Optional, TypeVar
import ray
from ray.data._internal.block_batching.interfaces import BlockPrefetcher
from ray.data._internal.block_batching.util import ActorBlockPrefetcher, WaitBlockPrefetcher, blocks_to_batches, collate, extract_data_from_batch, format_batches, resolve_block_refs
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats
from ray.data.block import Block, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
T = TypeVar('T')

def batch_block_refs(block_refs: Iterator[ObjectRef[Block]], *, stats: Optional[DatasetStats]=None, prefetch_blocks: int=0, clear_block_after_read: bool=False, batch_size: Optional[int]=None, batch_format: str='default', drop_last: bool=False, collate_fn: Optional[Callable[[DataBatch], Any]]=None, shuffle_buffer_min_size: Optional[int]=None, shuffle_seed: Optional[int]=None, ensure_copy: bool=False) -> Iterator[DataBatch]:
    if False:
        for i in range(10):
            print('nop')
    'Create formatted batches of data from 1 or more block object references.\n\n    This takes a block iterator and creates batch_size batches, slicing,\n    unioning, shuffling, prefetching, and formatting blocks as needed.\n\n    This is used by both Dataset.iter_batches() and Dataset.map_batches().\n\n    Args:\n        block_refs: An iterator over block object references.\n        prefetch_blocks: The number of blocks to prefetch ahead of the\n            current block during the scan.\n        clear_block_after_read: Whether to clear the block from object store\n            manually (i.e. without waiting for Python\'s automatic GC) after it\n            is read. Doing so will reclaim memory faster and hence reduce the\n            memory footprint. However, the caller has to ensure the safety, i.e.\n            the block will never be accessed again.\n        batch_size: Record batch size, or None to let the system pick.\n        batch_format: The format in which to return each batch.\n            Specify "default" to use the current block format (promoting\n            Arrow to pandas automatically), "pandas" to\n            select ``pandas.DataFrame`` or "pyarrow" to select\n            ``pyarrow.Table``. Default is "default".\n        drop_last: Whether to drop the last batch if it\'s incomplete.\n        collate_fn: A function to apply to each data batch before returning it.\n        shuffle_buffer_min_size: If non-None, the data will be randomly shuffled using a\n            local in-memory shuffle buffer, and this value will serve as the minimum\n            number of rows that must be in the local in-memory shuffle buffer in order\n            to yield a batch.\n        shuffle_seed: The seed to use for the local random shuffle.\n        ensure_copy: Whether batches are always copied from the underlying base\n            blocks (not zero-copy views).\n\n    Returns:\n        An iterator over record batches.\n    '
    context = DataContext.get_current()
    if prefetch_blocks > 0 and context.actor_prefetcher_enabled and (not ray.util.client.ray.is_connected()):
        prefetcher = ActorBlockPrefetcher()
    else:
        prefetcher = WaitBlockPrefetcher()
    eager_free = clear_block_after_read and DataContext.get_current().eager_free
    block_iter = resolve_block_refs(_prefetch_blocks(block_ref_iter=block_refs, prefetcher=prefetcher, num_blocks_to_prefetch=prefetch_blocks, eager_free=eager_free), stats=stats)
    yield from batch_blocks(block_iter, stats=stats, batch_size=batch_size, batch_format=batch_format, drop_last=drop_last, collate_fn=collate_fn, shuffle_buffer_min_size=shuffle_buffer_min_size, shuffle_seed=shuffle_seed, ensure_copy=ensure_copy)

def batch_blocks(blocks: Iterator[Block], *, stats: Optional[DatasetStats]=None, batch_size: Optional[int]=None, batch_format: str='default', drop_last: bool=False, collate_fn: Optional[Callable[[DataBatch], DataBatch]]=None, shuffle_buffer_min_size: Optional[int]=None, shuffle_seed: Optional[int]=None, ensure_copy: bool=False) -> Iterator[DataBatch]:
    if False:
        print('Hello World!')
    'Create formatted batches of data from 1 or more blocks.\n\n    This is equivalent to batch_block_refs, except\n    it takes in an iterator consisting of already fetched blocks.\n    This means that this function does not support block prefetching.\n    '

    def _iterator_fn(base_iterator: Iterator[Block]) -> Iterator[DataBatch]:
        if False:
            return 10
        batch_iter = format_batches(blocks_to_batches(block_iter=base_iterator, stats=stats, batch_size=batch_size, drop_last=drop_last, shuffle_buffer_min_size=shuffle_buffer_min_size, shuffle_seed=shuffle_seed, ensure_copy=ensure_copy), batch_format=batch_format, stats=stats)
        if collate_fn is not None:
            batch_iter = collate(batch_iter, collate_fn=collate_fn, stats=stats)
        batch_iter = extract_data_from_batch(batch_iter)
        yield from batch_iter
    batch_iter = _iterator_fn(blocks)
    for formatted_batch in batch_iter:
        user_timer = stats.iter_user_s.timer() if stats else nullcontext()
        with user_timer:
            yield formatted_batch

def _prefetch_blocks(block_ref_iter: Iterator[ObjectRef[Block]], prefetcher: BlockPrefetcher, num_blocks_to_prefetch: int, eager_free: bool=False, stats: Optional[DatasetStats]=None) -> Iterator[ObjectRef[Block]]:
    if False:
        i = 10
        return i + 15
    'Given an iterable of Block Object References, returns an iterator\n    over these object reference while prefetching `num_block_to_prefetch`\n    blocks in advance.\n\n    Args:\n        block_ref_iter: An iterator over block object references.\n        num_blocks_to_prefetch: The number of blocks to prefetch ahead of the\n            current block during the scan.\n        stats: Dataset stats object used to store block wait time.\n    '
    if num_blocks_to_prefetch == 0:
        for block_ref in block_ref_iter:
            yield block_ref
            trace_deallocation(block_ref, 'block_batching._prefetch_blocks', free=eager_free)
    window_size = num_blocks_to_prefetch
    sliding_window = collections.deque(itertools.islice(block_ref_iter, window_size), maxlen=window_size)
    with stats.iter_wait_s.timer() if stats else nullcontext():
        prefetcher.prefetch_blocks(list(sliding_window))
    while sliding_window:
        block_ref = sliding_window.popleft()
        try:
            sliding_window.append(next(block_ref_iter))
            with stats.iter_wait_s.timer() if stats else nullcontext():
                prefetcher.prefetch_blocks(list(sliding_window))
        except StopIteration:
            pass
        yield block_ref
        trace_deallocation(block_ref, 'block_batching._prefetch_blocks', free=eager_free)
    prefetcher.stop()