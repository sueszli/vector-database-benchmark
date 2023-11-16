import time
import numpy as np
import pandas as pd
import pytest
import ray
from ray._private.internal_api import memory_summary
from ray.data.block import BlockMetadata
from ray.data.datasource import Datasource, ReadTask
from ray.data.tests.conftest import *
from ray.tests.conftest import *

def test_input_backpressure_e2e(restore_data_context, shutdown_only):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class Counter:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.count = 0

        def increment(self):
            if False:
                for i in range(10):
                    print('nop')
            self.count += 1

        def get(self):
            if False:
                return 10
            return self.count

        def reset(self):
            if False:
                return 10
            self.count = 0

    class CountingRangeDatasource(Datasource):

        def __init__(self):
            if False:
                return 10
            self.counter = Counter.remote()

        def prepare_read(self, parallelism, n):
            if False:
                while True:
                    i = 10

            def range_(i):
                if False:
                    print('Hello World!')
                ray.get(self.counter.increment.remote())
                return [pd.DataFrame({'data': np.ones((n // parallelism * 1024 * 1024,))})]
            sz = n // parallelism * 1024 * 1024 * 8
            print('Block size', sz)
            return [ReadTask(lambda i=i: range_(i), BlockMetadata(num_rows=n // parallelism, size_bytes=sz, schema=None, input_files=None, exec_stats=None)) for i in range(parallelism)]
    source = CountingRangeDatasource()
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.resource_limits.object_store_memory = 10000000.0
    ds = ray.data.read_datasource(source, n=10000, parallelism=1000)
    it = iter(ds.iter_batches(batch_size=None, prefetch_batches=0))
    next(it)
    time.sleep(3)
    launched = ray.get(source.counter.get.remote())
    assert launched < 5, launched

def test_streaming_backpressure_e2e(restore_data_context):
    if False:
        print('Hello World!')

    class TestSlow:

        def __call__(self, df: np.ndarray):
            if False:
                while True:
                    i = 10
            time.sleep(2)
            return {'id': np.random.randn(1, 20, 1024, 1024)}

    class TestFast:

        def __call__(self, df: np.ndarray):
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(0.5)
            return {'id': np.random.randn(1, 20, 1024, 1024)}
    ctx = ray.init(object_store_memory=4000000000.0)
    ds = ray.data.range_tensor(20, shape=(3, 1024, 1024), parallelism=20)
    pipe = ds.map_batches(TestFast, batch_size=1, num_cpus=0.5, compute=ray.data.ActorPoolStrategy(size=2)).map_batches(TestSlow, batch_size=1, compute=ray.data.ActorPoolStrategy(size=1))
    for batch in pipe.iter_batches(batch_size=1, prefetch_batches=2):
        ...
    meminfo = memory_summary(ctx.address_info['address'], stats_only=True)
    assert 'Spilled' not in meminfo, meminfo
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))