import pytest
import ray
from ray.data import Dataset
from ray.data.context import DataContext
from ray.data.tests.conftest import *
from ray.tests.conftest import *

def test_map(ray_start_regular_shared, restore_data_context):
    if False:
        return 10
    ctx = DataContext.get_current()
    ctx.target_min_block_size = 100 * 8
    num_blocks_expected = 20
    ctx.target_max_block_size = 100 * 8
    ds = ray.data.range(1000).map(lambda row: row)
    assert ds.materialize().num_blocks() == num_blocks_expected
    ds = ray.data.range(1000).map(lambda row: row).map(lambda row: row)
    assert ds.materialize().num_blocks() == num_blocks_expected
    ctx.target_max_block_size *= 2
    ds = ray.data.range(1000).map(lambda row: row)
    assert ds.materialize().num_blocks() * 2 == num_blocks_expected
    ds = ray.data.range(1000).map(lambda row: row).map(lambda row: row)
    assert ds.materialize().num_blocks() * 2 == num_blocks_expected
    ctx.target_shuffle_max_block_size = 100 * 8
    ds = ray.data.range(1000).map(lambda row: row)
    assert ds.materialize().num_blocks() * 2 == num_blocks_expected
    ds = ray.data.range(1000).map(lambda row: row).map(lambda row: row)
    assert ds.materialize().num_blocks() * 2 == num_blocks_expected
SHUFFLE_ALL_TO_ALL_OPS = [(Dataset.random_shuffle, {}, True), (Dataset.sort, {'key': 'id'}, False)]

@pytest.mark.parametrize('shuffle_op', SHUFFLE_ALL_TO_ALL_OPS)
def test_shuffle(ray_start_regular_shared, restore_data_context, shuffle_op):
    if False:
        while True:
            i = 10
    ctx = DataContext.get_current()
    ctx.min_parallelism = 1
    ctx.target_min_block_size = 1
    mem_size = 8000
    (shuffle_fn, kwargs, fusion_supported) = shuffle_op
    ctx.target_shuffle_max_block_size = 100 * 8
    num_blocks_expected = mem_size // ctx.target_shuffle_max_block_size
    ds = shuffle_fn(ray.data.range(1000).map(lambda x: x), **kwargs)
    if not fusion_supported:
        num_blocks_expected *= 2
    assert ds.materialize().num_blocks() == num_blocks_expected
    ctx.target_shuffle_max_block_size /= 2
    num_blocks_expected = mem_size // ctx.target_shuffle_max_block_size
    ds = shuffle_fn(ray.data.range(1000), **kwargs)
    assert ds.materialize().num_blocks() == num_blocks_expected
    ds = shuffle_fn(ray.data.range(1000).map(lambda x: x), **kwargs)
    if not fusion_supported:
        num_blocks_expected *= 2
    assert ds.materialize().num_blocks() == num_blocks_expected
    ctx.target_max_block_size = 200 * 8
    num_blocks_expected = mem_size // ctx.target_shuffle_max_block_size
    ds = shuffle_fn(ray.data.range(1000), **kwargs)
    assert ds.materialize().num_blocks() == num_blocks_expected
    if not fusion_supported:
        num_blocks_expected *= 2
    ds = shuffle_fn(ray.data.range(1000).map(lambda x: x), **kwargs)
    assert ds.materialize().num_blocks() == num_blocks_expected
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))