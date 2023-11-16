import threading
import pytest
import ray
from ray._private.internal_api import memory_summary
from ray._private.test_utils import wait_for_condition
from ray.tests.conftest import *

def check_no_spill(ctx, dataset):
    if False:
        for i in range(10):
            print('nop')
    max_epoch = 10
    for _ in range(max_epoch):
        for _ in dataset.iter_batches(batch_size=None):
            pass
    meminfo = memory_summary(ctx.address_info['address'], stats_only=True)
    assert 'Spilled' not in meminfo, meminfo

    def _all_executor_threads_exited():
        if False:
            for i in range(10):
                print('nop')
        for thread in threading.enumerate():
            if thread.name.startswith('StreamingExecutor-'):
                return False
        return True
    wait_for_condition(_all_executor_threads_exited, timeout=10, retry_interval_ms=1000)

def check_to_torch_no_spill(ctx, dataset):
    if False:
        return 10
    max_epoch = 10
    for _ in range(max_epoch):
        for _ in dataset.to_torch(batch_size=None):
            pass
    meminfo = memory_summary(ctx.address_info['address'], stats_only=True)
    assert 'Spilled' not in meminfo, meminfo

def check_iter_torch_batches_no_spill(ctx, dataset):
    if False:
        i = 10
        return i + 15
    max_epoch = 10
    for _ in range(max_epoch):
        for _ in dataset.iter_torch_batches(batch_size=None):
            pass
    meminfo = memory_summary(ctx.address_info['address'], stats_only=True)
    assert 'Spilled' not in meminfo, meminfo

def check_to_tf_no_spill(ctx, dataset):
    if False:
        while True:
            i = 10
    max_epoch = 10
    for _ in range(max_epoch):
        for _ in dataset.to_tf(feature_columns='data', label_columns='label', batch_size=None):
            pass
    meminfo = memory_summary(ctx.address_info['address'], stats_only=True)
    assert 'Spilled' not in meminfo, meminfo

def check_iter_tf_batches_no_spill(ctx, dataset):
    if False:
        i = 10
        return i + 15
    max_epoch = 10
    for _ in range(max_epoch):
        for _ in dataset.iter_tf_batches():
            pass
    meminfo = memory_summary(ctx.address_info['address'], stats_only=True)
    assert 'Spilled' not in meminfo, meminfo

def test_iter_batches_no_spilling_upon_no_transformation(shutdown_only):
    if False:
        while True:
            i = 10
    ctx = ray.init(num_cpus=1, object_store_memory=300000000.0)
    ds = ray.data.range_tensor(500, shape=(80, 80, 4), parallelism=100)
    check_no_spill(ctx, ds)

def test_torch_iteration(shutdown_only):
    if False:
        i = 10
        return i + 15
    ctx = ray.init(num_cpus=1, object_store_memory=400000000.0)
    ds = ray.data.range_tensor(500, shape=(80, 80, 4), parallelism=100)
    check_to_torch_no_spill(ctx, ds)
    check_iter_torch_batches_no_spill(ctx, ds)

def test_tf_iteration(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ctx = ray.init(num_cpus=1, object_store_memory=800000000.0)
    ds = ray.data.range_tensor(500, shape=(80, 80, 4), parallelism=100).add_column('label', lambda x: 1)
    check_to_tf_no_spill(ctx, ds.map(lambda x: x))
    check_iter_tf_batches_no_spill(ctx, ds.map(lambda x: x))

def test_iter_batches_no_spilling_upon_prior_transformation(shutdown_only):
    if False:
        print('Hello World!')
    ctx = ray.init(num_cpus=1, object_store_memory=500000000.0)
    ds = ray.data.range_tensor(500, shape=(80, 80, 4), parallelism=100)
    check_no_spill(ctx, ds.map_batches(lambda x: x))

def test_iter_batches_no_spilling_upon_post_transformation(shutdown_only):
    if False:
        i = 10
        return i + 15
    ctx = ray.init(num_cpus=1, object_store_memory=500000000.0)
    ds = ray.data.range_tensor(500, shape=(80, 80, 4), parallelism=100)
    check_no_spill(ctx, ds.map_batches(lambda x: x, batch_size=5))

def test_iter_batches_no_spilling_upon_transformations(shutdown_only):
    if False:
        return 10
    ctx = ray.init(num_cpus=1, object_store_memory=700000000.0)
    ds = ray.data.range_tensor(500, shape=(80, 80, 4), parallelism=100)
    check_no_spill(ctx, ds.map_batches(lambda x: x).map_batches(lambda x: x))

def test_global_bytes_spilled(shutdown_only):
    if False:
        i = 10
        return i + 15
    ctx = ray.init(object_store_memory=90000000.0)
    ds = ray.data.range_tensor(500, shape=(80, 80, 4), parallelism=100).materialize().map_batches(lambda x: x).materialize()
    with pytest.raises(AssertionError):
        check_no_spill(ctx, ds)
    assert ds._get_stats_summary().global_bytes_spilled > 0
    assert ds._get_stats_summary().global_bytes_restored > 0
    assert 'Spilled to disk:' in ds.stats()

def test_no_global_bytes_spilled(shutdown_only):
    if False:
        while True:
            i = 10
    ctx = ray.init(object_store_memory=200000000.0)
    ds = ray.data.range_tensor(500, shape=(80, 80, 4), parallelism=100).materialize()
    check_no_spill(ctx, ds)
    assert ds._get_stats_summary().global_bytes_spilled == 0
    assert ds._get_stats_summary().global_bytes_restored == 0
    assert 'Cluster memory:' not in ds.stats()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))