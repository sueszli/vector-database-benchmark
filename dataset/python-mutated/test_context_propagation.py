import pandas as pd
import pytest
import ray
from ray._private.test_utils import run_string_as_driver
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask
from ray.tests.conftest import *

def test_context_saved_when_dataset_created(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    ctx = DataContext.get_current()
    d1 = ray.data.range(10)
    d2 = ray.data.range(10)
    assert ctx.eager_free
    assert d1.context.eager_free
    assert d2.context.eager_free
    d1.context.eager_free = False
    assert not d1.context.eager_free
    assert d2.context.eager_free
    assert ctx.eager_free

    @ray.remote(num_cpus=0)
    def check(d1, d2):
        if False:
            return 10
        assert not d1.context.eager_free
        assert d2.context.eager_free
    ray.get(check.remote(d1, d2))

    @ray.remote(num_cpus=0)
    def check2(d):
        if False:
            while True:
                i = 10
        d.take()

    @ray.remote(num_cpus=0)
    def check3(d):
        if False:
            while True:
                i = 10
        list(d.streaming_split(1)[0].iter_batches())
    d1.context.execution_options.resource_limits.cpu = 0.1
    ray.get(check2.remote(d2))
    ray.get(check3.remote(d2))

def test_read(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    class CustomDatasource(Datasource):

        def prepare_read(self, parallelism: int):
            if False:
                return 10
            value = DataContext.get_current().foo
            meta = BlockMetadata(num_rows=1, size_bytes=8, schema=None, input_files=None, exec_stats=None)
            return [ReadTask(lambda : [pd.DataFrame({'id': [value]})], meta)]
    context = DataContext.get_current()
    context.foo = 12345
    assert ray.data.read_datasource(CustomDatasource()).take_all()[0]['id'] == 12345

def test_map(ray_start_regular_shared):
    if False:
        while True:
            i = 10
    context = DataContext.get_current()
    context.foo = 70001
    ds = ray.data.range(1).map(lambda x: {'id': DataContext.get_current().foo})
    assert ds.take_all()[0]['id'] == 70001

def test_flat_map(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15
    context = DataContext.get_current()
    context.foo = 70002
    ds = ray.data.range(1).flat_map(lambda x: [{'id': DataContext.get_current().foo}])
    assert ds.take_all()[0]['id'] == 70002

def test_map_batches(ray_start_regular_shared):
    if False:
        return 10
    context = DataContext.get_current()
    context.foo = 70003
    ds = ray.data.range(1).map_batches(lambda x: {'id': [DataContext.get_current().foo]})
    assert ds.take_all()[0]['id'] == 70003

def test_filter(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    context = DataContext.get_current()
    context.foo = 70004
    ds = ray.data.from_items([70004]).filter(lambda x: x['item'] == DataContext.get_current().foo)
    assert ds.take_all()[0]['item'] == 70004

def test_context_placement_group():
    if False:
        i = 10
        return i + 15
    driver_code = '\nimport ray\nfrom ray.data.context import DataContext\nfrom ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy\nfrom ray._private.test_utils import placement_group_assert_no_leak\n\nray.init(num_cpus=1)\n\ncontext = DataContext.get_current()\n# This placement group will take up all cores of the local cluster.\nplacement_group = ray.util.placement_group(\n    name="core_hog",\n    strategy="SPREAD",\n    bundles=[\n        {"CPU": 1},\n    ],\n)\nray.get(placement_group.ready())\ncontext.scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group)\nds = ray.data.range(100, parallelism=2).map(lambda x: {"id": x["id"] + 1})\nassert ds.take_all() == [{"id": x} for x in range(1, 101)]\nplacement_group_assert_no_leak([placement_group])\nray.shutdown()\n    '
    run_string_as_driver(driver_code)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))