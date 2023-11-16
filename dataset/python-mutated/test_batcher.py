import time
import pyarrow as pa
import pytest
import ray
from ray.data._internal.batcher import Batcher, ShufflingBatcher

def gen_block(num_rows):
    if False:
        return 10
    return pa.table({'foo': [1] * num_rows})

def test_shuffling_batcher():
    if False:
        return 10
    batch_size = 5
    buffer_size = 20
    with pytest.raises(ValueError, match='Must specify a batch_size if using a local shuffle.'):
        ShufflingBatcher(batch_size=None, shuffle_buffer_min_size=buffer_size)
    ShufflingBatcher(batch_size=batch_size, shuffle_buffer_min_size=batch_size - 1)
    batcher = ShufflingBatcher(batch_size=batch_size, shuffle_buffer_min_size=buffer_size)

    def add_and_check(num_rows, materialized_buffer_size, pending_buffer_size, expect_has_batch=False, no_nexting_yet=True):
        if False:
            while True:
                i = 10
        block = gen_block(num_rows)
        batcher.add(block)
        if expect_has_batch:
            assert batcher.has_batch()
        else:
            assert not batcher.has_batch()
        if no_nexting_yet:
            assert batcher._shuffle_buffer is None
            assert batcher._batch_head == 0
        assert batcher._builder.num_rows() == pending_buffer_size
        assert batcher._materialized_buffer_size() == materialized_buffer_size

    def next_and_check(current_cursor, materialized_buffer_size, pending_buffer_size, should_batch_be_full=True, should_have_batch_after=True, new_data_added=False):
        if False:
            print('Hello World!')
        if should_batch_be_full:
            assert batcher.has_batch()
        else:
            batcher.has_any()
        if new_data_added:
            assert batcher._builder.num_rows() > 0
        batch = batcher.next_batch()
        if should_batch_be_full:
            assert len(batch) == batch_size
        assert batcher._builder.num_rows() == pending_buffer_size
        assert batcher._materialized_buffer_size() == materialized_buffer_size
        if should_have_batch_after:
            assert batcher.has_batch()
        else:
            assert not batcher.has_batch()
    add_and_check(3, materialized_buffer_size=0, pending_buffer_size=3)
    add_and_check(7, materialized_buffer_size=0, pending_buffer_size=10)
    add_and_check(10, materialized_buffer_size=0, pending_buffer_size=20)
    add_and_check(15, materialized_buffer_size=0, pending_buffer_size=35, expect_has_batch=True)
    next_and_check(0, materialized_buffer_size=30, pending_buffer_size=0, should_have_batch_after=True, new_data_added=True)
    add_and_check(20, materialized_buffer_size=30, pending_buffer_size=20, expect_has_batch=True, no_nexting_yet=False)
    next_and_check(0, materialized_buffer_size=25, pending_buffer_size=20, new_data_added=True)
    next_and_check(batch_size, materialized_buffer_size=20, pending_buffer_size=20)
    next_and_check(2 * batch_size, materialized_buffer_size=35, pending_buffer_size=0)
    next_and_check(3 * batch_size, materialized_buffer_size=30, pending_buffer_size=0, should_have_batch_after=True)
    add_and_check(8, materialized_buffer_size=30, pending_buffer_size=8, expect_has_batch=True, no_nexting_yet=False)
    next_and_check(0, materialized_buffer_size=25, pending_buffer_size=8, should_have_batch_after=True, new_data_added=True)
    batcher.done_adding()
    next_and_check(batch_size, materialized_buffer_size=28, pending_buffer_size=0)
    next_and_check(2 * batch_size, materialized_buffer_size=23, pending_buffer_size=0)
    next_and_check(3 * batch_size, materialized_buffer_size=18, pending_buffer_size=0)
    next_and_check(4 * batch_size, materialized_buffer_size=13, pending_buffer_size=0)
    next_and_check(5 * batch_size, materialized_buffer_size=8, pending_buffer_size=0)
    next_and_check(6 * batch_size, materialized_buffer_size=3, pending_buffer_size=0, should_have_batch_after=False)
    next_and_check(7 * batch_size, materialized_buffer_size=0, pending_buffer_size=0, should_batch_be_full=False, should_have_batch_after=False)

def test_batching_pyarrow_table_with_many_chunks():
    if False:
        for i in range(10):
            print('nop')
    'Make sure batching a pyarrow table with many chunks is fast.\n\n    See https://github.com/ray-project/ray/issues/31108 for more details.\n    '
    num_chunks = 5000
    batch_size = 1024
    batches = []
    for _ in range(num_chunks):
        batch = {}
        for i in range(10):
            batch[str(i)] = list(range(batch_size))
        batches.append(pa.Table.from_pydict(batch))
    block = pa.concat_tables(batches, promote=True)
    start = time.perf_counter()
    batcher = Batcher(batch_size, ensure_copy=False)
    batcher.add(block)
    batcher.done_adding()
    while batcher.has_any():
        batcher.next_batch()
    duration = time.perf_counter() - start
    assert duration < 10
    start = time.perf_counter()
    shuffling_batcher = ShufflingBatcher(batch_size=batch_size, shuffle_buffer_min_size=batch_size)
    shuffling_batcher.add(block)
    shuffling_batcher.done_adding()
    while shuffling_batcher.has_any():
        shuffling_batcher.next_batch()
    duration = time.perf_counter() - start
    assert duration < 30

@pytest.mark.parametrize('batch_size,local_shuffle_buffer_size', [(1, 1), (10, 1), (1, 10), (10, 1000), (1000, 10)])
def test_shuffling_batcher_grid(batch_size, local_shuffle_buffer_size):
    if False:
        while True:
            i = 10
    ds = ray.data.range_tensor(10000, shape=(130,))
    start = time.time()
    count = 0
    for batch in ds.iter_batches(batch_size=batch_size, local_shuffle_buffer_size=local_shuffle_buffer_size):
        count += len(batch['data'])
    print(ds.size_bytes() / 1000000000.0 / (time.time() - start), 'GB/s')
    assert count == 10000
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))