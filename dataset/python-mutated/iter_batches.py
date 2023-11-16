import collections
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import ray
from ray.data._internal.block_batching.interfaces import Batch, BlockPrefetcher
from ray.data._internal.block_batching.util import ActorBlockPrefetcher, WaitBlockPrefetcher, blocks_to_batches, collate, extract_data_from_batch, finalize_batches, format_batches, resolve_block_refs
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats, clear_stats_actor_iter_metrics, update_stats_actor_iter_metrics
from ray.data._internal.util import make_async_gen
from ray.data.block import Block, BlockMetadata, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
STATS_UPDATE_INTERVAL_SECONDS = 30

def iter_batches(block_refs: Iterator[Tuple[ObjectRef[Block], BlockMetadata]], dataset_tag: str, *, stats: Optional[DatasetStats]=None, clear_block_after_read: bool=False, batch_size: Optional[int]=None, batch_format: Optional[str]='default', drop_last: bool=False, collate_fn: Optional[Callable[[DataBatch], Any]]=None, finalize_fn: Optional[Callable[[Any], Any]]=None, shuffle_buffer_min_size: Optional[int]=None, shuffle_seed: Optional[int]=None, ensure_copy: bool=False, prefetch_batches: int=1) -> Iterator[DataBatch]:
    if False:
        return 10
    'Create formatted batches of data from an iterator of block object references and\n    corresponding metadata.\n\n    This takes a block iterator and creates batch_size batches, slicing,\n    unioning, shuffling, prefetching, and formatting blocks as needed.\n\n    The algorithm uses both pipeline parallelism and data parallelism:\n\n    If prefetch_batches=2, these are all the batches in flight:\n\n    [User thread] trains on Batch 0\n    - [Fetch thread] Batch 1 finalization + move to output queue\n            - [Worker thread 1] Batch 2 formatting + collating\n            - [Worker thread 2] Batch 3 formatting + collating\n            - [Raylet] Batches 4 + 5 fetched to local object store memory\n\n    At any point in time there are prefetch_batches+1 batches in local heap memory.\n    And the next set of prefetch_batches in local object store memory.\n\n    The actual steps are as follows:\n\n    In a single async thread, do the following:\n        1. Trigger Ray local prefetching of `prefetch_batches` worth of block object\n            references.\n        2. Resolve (i.e. call `ray.get()`) on the block references.\n        3. Perform the necessary batch slicing to construct full batches, possibly\n            shuffling if necessary.\n        4. Then, in a threadpool consisting of `prefetch_batches` threads:\n            a. Format the batches to the provided batch format.\n            b. Apply the collate function.\n        5. Finalize each of the collated batches\n        6. Fetch outputs from the threadpool, maintaining order of the batches.\n\n    Args:\n        block_refs: An iterator over block object references and their corresponding\n            metadata.\n        stats: DatasetStats object to record timing and other statistics.\n        clear_block_after_read: Whether to clear the block from object store\n            manually (i.e. without waiting for Python\'s automatic GC) after it\n            is read. Doing so will reclaim memory faster and hence reduce the\n            memory footprint. However, the caller has to ensure the safety, i.e.\n            the block will never be accessed again.\n        batch_size: Record batch size, or None to let the system pick.\n        batch_format: The format in which to return each batch.\n            Specify "default" to use the current block format (promoting\n            Arrow to pandas automatically), "pandas" to\n            select ``pandas.DataFrame`` or "pyarrow" to select\n            ``pyarrow.Table``, or None to use entire blocks\n            as batches. Default is "default".\n        drop_last: Whether to drop the last batch if it\'s incomplete.\n        collate_fn: A function to apply to each data batch before returning it.\n        finalize_fn: A function to apply to each data batch after it has been collated.\n            This function is not run in a threadpool so it can be used for\n            memory-intensive operations such as GPU preloading.\n        shuffle_buffer_min_size: If non-None, the data will be randomly shuffled using a\n            local in-memory shuffle buffer, and this value will serve as the minimum\n            number of rows that must be in the local in-memory shuffle buffer in order\n            to yield a batch.\n        shuffle_seed: The seed to use for the local random shuffle.\n        ensure_copy: Whether batches are always copied from the underlying base\n            blocks (not zero-copy views).\n        prefetch_batches: The number of batches to fetch ahead of the current batch to\n            process. If set to greater than 0, a separate thread will be used to fetch\n            the specified amount of formatted batches from blocks. This improves\n            performance for non-CPU bound UDFs, allowing batch fetching compute and\n            formatting to be overlapped with the UDF. Defaults to 1.\n\n    Returns:\n        An iterator over record batches.\n    '
    context = DataContext.get_current()
    if prefetch_batches > 0 and context.actor_prefetcher_enabled and (not ray.util.client.ray.is_connected()):
        prefetcher = ActorBlockPrefetcher()
    else:
        prefetcher = WaitBlockPrefetcher()
    eager_free = clear_block_after_read and DataContext.get_current().eager_free

    def _async_iter_batches(block_refs: Iterator[Tuple[ObjectRef[Block], BlockMetadata]]) -> Iterator[DataBatch]:
        if False:
            while True:
                i = 10
        block_refs = prefetch_batches_locally(block_ref_iter=block_refs, prefetcher=prefetcher, num_batches_to_prefetch=prefetch_batches, batch_size=batch_size, eager_free=eager_free)
        block_iter = resolve_block_refs(block_ref_iter=block_refs, stats=stats)
        batch_iter = blocks_to_batches(block_iter=block_iter, stats=stats, batch_size=batch_size, drop_last=drop_last, shuffle_buffer_min_size=shuffle_buffer_min_size, shuffle_seed=shuffle_seed, ensure_copy=ensure_copy)
        batch_iter = _format_in_threadpool(batch_iter, stats=stats, batch_format=batch_format, collate_fn=collate_fn, num_threadpool_workers=prefetch_batches)
        if finalize_fn is not None:
            batch_iter = finalize_batches(batch_iter, finalize_fn=finalize_fn, stats=stats)
        batch_iter: Iterator[Batch] = restore_original_order(batch_iter)
        yield from extract_data_from_batch(batch_iter)
    async_batch_iter = make_async_gen(block_refs, fn=_async_iter_batches, num_workers=1)
    metrics_tag = {'dataset': dataset_tag}
    last_stats_update_time = 0
    while True:
        with stats.iter_total_blocked_s.timer() if stats else nullcontext():
            try:
                next_batch = next(async_batch_iter)
            except StopIteration:
                break
        with stats.iter_user_s.timer() if stats else nullcontext():
            yield next_batch
        if time.time() - last_stats_update_time >= STATS_UPDATE_INTERVAL_SECONDS:
            update_stats_actor_iter_metrics(stats, metrics_tag)
            last_stats_update_time = time.time()
    clear_stats_actor_iter_metrics(metrics_tag)

def _format_in_threadpool(batch_iter: Iterator[Batch], stats: DatasetStats, batch_format: Optional[str], collate_fn: Optional[Callable[[DataBatch], Any]], num_threadpool_workers: int) -> Iterator[Batch]:
    if False:
        return 10
    'Executes the batching, formatting, and collation logic in a threadpool.\n\n    Args:\n        logical_batch_iterator: An iterator over logical batches.\n        stats: DatasetStats object to record timing and other statistics.\n        batch_format: The format in which to return each batch.\n            Specify "default" to use the current block format (promoting\n            Arrow to pandas automatically), "pandas" to\n            select ``pandas.DataFrame`` or "pyarrow" to select\n            ``pyarrow.Table``, or None to use entire blocks\n            as batches.\n        collate_fn: A function to apply to each data batch before returning it.\n        num_threadpool_workers: The number of threads to use in the threadpool.\n    '

    def threadpool_computations_format_collate(batch_iter: Iterator[Batch]) -> Iterator[Batch]:
        if False:
            while True:
                i = 10
        formatted_batch_iter = format_batches(batch_iter, batch_format=batch_format, stats=stats)
        if collate_fn is not None:
            formatted_batch_iter = collate(formatted_batch_iter, collate_fn=collate_fn, stats=stats)
        yield from formatted_batch_iter
    if num_threadpool_workers > 0:
        collated_iter = make_async_gen(base_iterator=batch_iter, fn=threadpool_computations_format_collate, num_workers=num_threadpool_workers)
    else:
        collated_iter = threadpool_computations_format_collate(batch_iter)
    return collated_iter

def prefetch_batches_locally(block_ref_iter: Iterator[Tuple[ObjectRef[Block], BlockMetadata]], prefetcher: BlockPrefetcher, num_batches_to_prefetch: int, batch_size: Optional[int], eager_free: bool=False) -> Iterator[ObjectRef[Block]]:
    if False:
        return 10
    'Given an iterator of batched block references, returns an iterator over the same\n    block references while prefetching `num_batches_to_prefetch` batches in advance.\n\n    Args:\n        block_ref_iter: An iterator over batched block references.\n        prefetcher: The prefetcher to use.\n        num_batches_to_prefetch: The number of batches to prefetch ahead of the\n            current batch during the scan.\n        batch_size: User specified batch size, or None to let the system pick.\n        eager_free: Whether to eagerly free the object reference from the object store.\n    '
    sliding_window = collections.deque()
    current_window_size = 0
    if num_batches_to_prefetch <= 0:
        for (block_ref, metadata) in block_ref_iter:
            yield block_ref
        return
    if batch_size is not None:
        num_rows_to_prefetch = num_batches_to_prefetch * batch_size
    else:
        num_rows_to_prefetch = None
    while batch_size is not None and current_window_size < num_rows_to_prefetch or (batch_size is None and len(sliding_window) < num_batches_to_prefetch):
        try:
            next_block_ref_and_metadata = next(block_ref_iter)
        except StopIteration:
            break
        sliding_window.append(next_block_ref_and_metadata)
        current_window_size += next_block_ref_and_metadata[1].num_rows
    prefetcher.prefetch_blocks([block_ref for (block_ref, _) in list(sliding_window)])
    while sliding_window:
        (block_ref, metadata) = sliding_window.popleft()
        current_window_size -= metadata.num_rows
        if batch_size is None or current_window_size < num_rows_to_prefetch:
            try:
                sliding_window.append(next(block_ref_iter))
                prefetcher.prefetch_blocks([block_ref for (block_ref, _) in list(sliding_window)])
            except StopIteration:
                pass
        yield block_ref
        trace_deallocation(block_ref, loc='iter_batches', free=eager_free)
    prefetcher.stop()

def restore_original_order(batch_iter: Iterator[Batch]) -> Iterator[Batch]:
    if False:
        return 10
    "Restores the original order of the provided `batch_iter`\n\n    This function will yield items from `base_iterator` in the correct order based on\n    each batch's batch_idx. All indexes are expected to be unique.\n\n    `batch_iter` is expected to not have any missing indexes. All indexes from 0 to len\n    (base_iterator) must be present.\n    "
    next_index_required = 0
    buffer: Dict[int, Batch] = {}
    for batch in batch_iter:
        assert batch.batch_idx not in buffer
        buffer[batch.batch_idx] = batch
        while next_index_required in buffer:
            yield buffer.pop(next_index_required)
            next_index_required += 1
    while next_index_required in buffer:
        yield buffer.pop(next_index_required)
        next_index_required += 1