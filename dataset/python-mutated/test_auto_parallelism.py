from dataclasses import astuple, dataclass
import pytest
import ray
from ray.data._internal.util import _autodetect_parallelism
from ray.data.context import DataContext
from ray.tests.conftest import *
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@dataclass
class TestCase:
    avail_cpus: int
    target_max_block_size: int
    data_size: int
    expected_parallelism: int
MiB = 1024 * 1024
GiB = 1024 * MiB
TEST_CASES = [TestCase(avail_cpus=4, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=1024, expected_parallelism=8), TestCase(avail_cpus=4, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=10 * MiB, expected_parallelism=10), TestCase(avail_cpus=4, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=20 * MiB, expected_parallelism=20), TestCase(avail_cpus=4, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=100 * MiB, expected_parallelism=100), TestCase(avail_cpus=4, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=1 * GiB, expected_parallelism=200), TestCase(avail_cpus=4, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=10 * GiB, expected_parallelism=200), TestCase(avail_cpus=150, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=10 * GiB, expected_parallelism=300), TestCase(avail_cpus=400, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=10 * GiB, expected_parallelism=800), TestCase(avail_cpus=400, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=1 * MiB, expected_parallelism=800), TestCase(avail_cpus=4, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=1000 * GiB, expected_parallelism=8000), TestCase(avail_cpus=4, target_max_block_size=DataContext.get_current().target_max_block_size, data_size=10000 * GiB, expected_parallelism=80000), TestCase(avail_cpus=4, target_max_block_size=512 * MiB, data_size=1000 * GiB, expected_parallelism=2000), TestCase(avail_cpus=4, target_max_block_size=512 * MiB, data_size=10000 * GiB, expected_parallelism=20000)]

@pytest.mark.parametrize('avail_cpus,target_max_block_size,data_size,expected', [astuple(test) for test in TEST_CASES])
def test_autodetect_parallelism(shutdown_only, avail_cpus, target_max_block_size, data_size, expected):
    if False:
        return 10

    class MockReader:

        def estimate_inmemory_data_size(self):
            if False:
                while True:
                    i = 10
            return data_size
    (result, _, _, _) = _autodetect_parallelism(parallelism=-1, target_max_block_size=target_max_block_size, ctx=DataContext.get_current(), datasource_or_legacy_reader=MockReader(), avail_cpus=avail_cpus)
    assert result == expected, (result, expected)

def test_auto_parallelism_basic(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init(num_cpus=8)
    context = DataContext.get_current()
    context.min_parallelism = 1
    ds = ray.data.range_tensor(5, shape=(100,), parallelism=-1)
    assert ds.num_blocks() == 5, ds
    ds = ray.data.range_tensor(10000, shape=(100,), parallelism=-1)
    assert ds.num_blocks() == 16, ds
    ds = ray.data.range_tensor(100000000, shape=(100,), parallelism=-1)
    assert ds.num_blocks() >= 590, ds
    assert ds.num_blocks() <= 600, ds

def test_auto_parallelism_placement_group(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=16, num_gpus=8)

    @ray.remote
    def run():
        if False:
            while True:
                i = 10
        context = DataContext.get_current()
        context.min_parallelism = 1
        ds = ray.data.range_tensor(2000, shape=(100,), parallelism=-1)
        return ds.num_blocks()
    pg = ray.util.placement_group([{'CPU': 1}])
    num_blocks = ray.get(run.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote())
    assert num_blocks == 4, num_blocks
    pg = ray.util.placement_group([{'CPU': 2}])
    num_blocks = ray.get(run.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote())
    assert num_blocks == 8, num_blocks
    pg = ray.util.placement_group([{'CPU': 1, 'GPU': 1}])
    num_blocks = ray.get(run.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote())
    assert num_blocks == 8, num_blocks
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))