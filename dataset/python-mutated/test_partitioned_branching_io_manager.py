import math
import time
from dagster import DagsterInstance, Definitions, asset
from dagster._core.definitions.assets import AssetsDefinition
from dagster._core.definitions.partition import StaticPartitionsDefinition
from dagster._core.storage.branching.branching_io_manager import BranchingIOManager
from .utils import AssetBasedInMemoryIOManager, DefinitionsRunner
partitioning_scheme = StaticPartitionsDefinition(['A', 'B', 'C'])

@asset(partitions_def=partitioning_scheme)
def now_time():
    if False:
        i = 10
        return i + 15
    return int(math.floor(time.time() * 100))

def get_now_time_plus_N(N: int) -> AssetsDefinition:
    if False:
        for i in range(10):
            print('nop')

    @asset(partitions_def=partitioning_scheme)
    def now_time_plus_N(now_time: int) -> int:
        if False:
            while True:
                i = 10
        return now_time + N
    return now_time_plus_N

@asset(partitions_def=partitioning_scheme)
def now_time_plus_20_after_plus_N(now_time_plus_N: int) -> int:
    if False:
        while True:
            i = 10
    return now_time_plus_N + 20

def test_asset_based_io_manager_with_partitions():
    if False:
        print('Hello World!')

    @asset(partitions_def=partitioning_scheme)
    def plus_10(now_time):
        if False:
            while True:
                i = 10
        return now_time + 10
    io_manager = AssetBasedInMemoryIOManager()
    with DefinitionsRunner.ephemeral(Definitions(assets=[now_time, plus_10], resources={'io_manager': io_manager})) as runner:
        partition_A_result = runner.materialize_all_assets(partition_key='A')
        partition_A_time_now = partition_A_result.output_for_node('now_time')
        assert isinstance(partition_A_time_now, int)
        assert not io_manager.has_value('now_time')
        assert io_manager.has_value('now_time', partition_key='A')
        assert not io_manager.has_value('now_time', partition_key='B')
        assert io_manager.get_value('now_time', partition_key='A') == partition_A_time_now
        partition_B_result = runner.materialize_all_assets(partition_key='B')
        assert partition_B_result.success
        partition_B_time_now = partition_B_result.output_for_node('now_time')
        assert isinstance(partition_B_time_now, int)
        assert partition_B_time_now > partition_A_time_now
        assert not io_manager.has_value('now_time')
        assert io_manager.has_value('now_time', partition_key='A')
        assert io_manager.has_value('now_time', partition_key='B')
        assert io_manager.get_value('now_time', partition_key='A') == partition_A_time_now
        assert io_manager.get_value('now_time', partition_key='B') == partition_B_time_now
        assert runner.load_asset_value('now_time', partition_key='A') == partition_A_time_now
        assert runner.load_asset_value('now_time', partition_key='B') == partition_B_time_now

def test_basic_partitioning_workflow():
    if False:
        print('Hello World!')
    now_time_plus_10 = get_now_time_plus_N(10)
    prod_io_manager = AssetBasedInMemoryIOManager()
    dev_io_manager = AssetBasedInMemoryIOManager()
    prod_defs = Definitions(assets=[now_time, now_time_plus_10, now_time_plus_20_after_plus_N], resources={'io_manager': prod_io_manager})
    dev_defs_t0 = Definitions(assets=[now_time, get_now_time_plus_N(10), now_time_plus_20_after_plus_N], resources={'io_manager': BranchingIOManager(parent_io_manager=prod_io_manager, branch_io_manager=dev_io_manager)})
    dev_defs_t1 = Definitions(assets=[now_time, get_now_time_plus_N(15), now_time_plus_20_after_plus_N], resources={'io_manager': BranchingIOManager(parent_io_manager=prod_io_manager, branch_io_manager=dev_io_manager)})
    with DagsterInstance.ephemeral() as dev_instance, DagsterInstance.ephemeral() as prod_instance:
        prod_runner = DefinitionsRunner(prod_defs, prod_instance)
        prod_runner.materialize_all_assets(partition_key='A')
        prod_runner.materialize_all_assets(partition_key='B')
        prod_runner.materialize_all_assets(partition_key='C')
        for asset_key in ['now_time', 'now_time_plus_N', 'now_time_plus_20_after_plus_N']:
            for partition_key in ['A', 'B', 'C']:
                assert prod_io_manager.has_value(asset_key, partition_key)
        prod_now_time_A = prod_runner.load_asset_value('now_time', partition_key='A')
        assert isinstance(prod_now_time_A, int)
        prod_now_time_B = prod_runner.load_asset_value('now_time', partition_key='B')
        assert isinstance(prod_now_time_B, int)
        dev_runner_t0 = DefinitionsRunner(dev_defs_t0, dev_instance)
        dev_runner_t0.materialize_asset('now_time_plus_N', partition_key='A')
        assert dev_runner_t0.load_asset_value('now_time', partition_key='A') == prod_runner.load_asset_value('now_time', partition_key='A')
        assert not dev_io_manager.has_value('now_time', partition_key='A')
        assert dev_runner_t0.load_asset_value('now_time_plus_N', partition_key='A') == prod_runner.load_asset_value('now_time_plus_N', partition_key='A')
        assert dev_runner_t0.load_asset_value('now_time_plus_N', partition_key='A') == prod_now_time_A + 10
        dev_runner_t1 = DefinitionsRunner(dev_defs_t1, dev_instance)
        dev_runner_t1.materialize_asset('now_time_plus_N', partition_key='A')
        assert dev_runner_t1.load_asset_value('now_time', partition_key='A') == prod_runner.load_asset_value('now_time', partition_key='A')
        assert dev_runner_t1.load_asset_value('now_time_plus_N', partition_key='A') != prod_runner.load_asset_value('now_time_plus_N', partition_key='A')
        assert dev_io_manager.get_value('now_time_plus_N', partition_key='A') != prod_io_manager.get_value('now_time_plus_N', partition_key='A')
        assert not dev_io_manager.has_value('now_time', partition_key='A')
        assert not dev_io_manager.has_value('now_time_plus_N', partition_key='B')
        assert not dev_io_manager.has_value('now_time_plus_N', partition_key='C')
        dev_runner_t1.materialize_asset('now_time_plus_20_after_plus_N', partition_key='A')
        assert dev_runner_t1.load_asset_value('now_time_plus_20_after_plus_N', partition_key='A') == prod_now_time_A + 15 + 20
        assert prod_runner.load_asset_value('now_time_plus_20_after_plus_N', partition_key='A') == prod_now_time_A + 10 + 20
        assert dev_runner_t1.load_asset_value('now_time_plus_N', partition_key='B') == prod_now_time_B + 10
        dev_runner_t1.materialize_asset('now_time', partition_key='B')
        assert dev_runner_t1.load_asset_value('now_time_plus_N', partition_key='B') == prod_now_time_B + 10
        dev_now_time_B = dev_runner_t1.load_asset_value('now_time', partition_key='B')
        assert isinstance(dev_now_time_B, int)
        assert dev_now_time_B > prod_now_time_B
        dev_runner_t1.materialize_asset('now_time_plus_N', partition_key='B')
        assert dev_runner_t1.load_asset_value('now_time_plus_N', partition_key='B') == dev_now_time_B + 15