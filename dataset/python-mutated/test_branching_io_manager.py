import math
import time
from typing import Optional
from dagster import Definitions, asset
from dagster._core.definitions.assets import AssetsDefinition
from dagster._core.definitions.events import AssetKey, AssetMaterialization
from dagster._core.definitions.metadata import TextMetadataValue
from dagster._core.events.log import EventLogEntry
from dagster._core.instance import DagsterInstance
from dagster._core.storage.branching.branching_io_manager import BranchingIOManager
from .utils import LOG, AssetBasedInMemoryIOManager, ConfigurableAssetBasedInMemoryIOManager, DefinitionsRunner

@asset
def now_time() -> int:
    if False:
        print('Hello World!')
    return int(math.floor(time.time() * 100))

def get_now_time_plus_N(N: int) -> AssetsDefinition:
    if False:
        print('Hello World!')

    @asset
    def now_time_plus_N(now_time: int) -> int:
        if False:
            print('Hello World!')
        return now_time + N
    return now_time_plus_N

@asset
def now_time_plus_20_after_plus_N(now_time_plus_N: int) -> int:
    if False:
        i = 10
        return i + 15
    return now_time_plus_N + 20

def test_basic_bound_runner_usage():
    if False:
        for i in range(10):
            print('nop')
    with DefinitionsRunner.ephemeral(Definitions(assets=[now_time], resources={'io_manager': AssetBasedInMemoryIOManager()})) as runner:
        assert runner.materialize_all_assets().success
        assert isinstance(runner.load_asset_value('now_time'), int)

def get_env_entry(event_log_entry: EventLogEntry, metadata_key='io_manager_branch') -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    asset_mat = event_log_entry.asset_materialization
    return get_branch_name_from_materialization(asset_mat, metadata_key=metadata_key) if asset_mat else None

def get_branch_name_from_materialization(asset_materialization: AssetMaterialization, metadata_key='io_manager_branch') -> Optional[str]:
    if False:
        return 10
    entry = asset_materialization.metadata.get(metadata_key)
    if isinstance(entry, TextMetadataValue):
        return entry.value
    else:
        return None

def test_write_staging_label():
    if False:
        return 10
    with DefinitionsRunner.ephemeral(Definitions(assets=[now_time], resources={'io_manager': BranchingIOManager(parent_io_manager=AssetBasedInMemoryIOManager(), branch_io_manager=AssetBasedInMemoryIOManager())})) as staging_runner:
        assert staging_runner.materialize_all_assets().success
        all_mat_log_records = staging_runner.get_all_asset_materialization_event_records('now_time')
        assert all_mat_log_records
        assert len(all_mat_log_records) == 1
        asset_mat = all_mat_log_records[0].event_log_entry.asset_materialization
        assert asset_mat
        assert get_branch_name_from_materialization(asset_mat) == 'dev'

def test_setup_teardown() -> None:
    if False:
        i = 10
        return i + 15
    with DefinitionsRunner.ephemeral(Definitions(assets=[now_time], resources={'io_manager': BranchingIOManager(parent_io_manager=ConfigurableAssetBasedInMemoryIOManager(name='parent'), branch_io_manager=ConfigurableAssetBasedInMemoryIOManager(name='branch'))})) as staging_runner:
        LOG.clear()
        assert staging_runner.materialize_all_assets().success
        assert len(LOG) == 4
        assert LOG == ['setup_for_execution parent', 'setup_for_execution branch', 'teardown_after_execution branch', 'teardown_after_execution parent']
        all_mat_log_records = staging_runner.get_all_asset_materialization_event_records('now_time')
        assert all_mat_log_records
        assert len(all_mat_log_records) == 1
        asset_mat = all_mat_log_records[0].event_log_entry.asset_materialization
        assert asset_mat
        assert get_branch_name_from_materialization(asset_mat) == 'dev'

def test_write_alternative_branch_metadata_key():
    if False:
        for i in range(10):
            print('nop')
    with DefinitionsRunner.ephemeral(Definitions(assets=[now_time], resources={'io_manager': BranchingIOManager(parent_io_manager=AssetBasedInMemoryIOManager(), branch_io_manager=AssetBasedInMemoryIOManager(), branch_metadata_key='another_key')})) as staging_runner:
        assert staging_runner.materialize_all_assets()
        all_mat_log_records = staging_runner.get_all_asset_materialization_event_records('now_time')
        assert all_mat_log_records
        assert len(all_mat_log_records) == 1
        asset_mat = all_mat_log_records[0].event_log_entry.asset_materialization
        assert asset_mat
        assert get_branch_name_from_materialization(asset_mat, metadata_key='another_key') == 'dev'

def test_basic_workflow():
    if False:
        i = 10
        return i + 15
    'In this test we are going to iterate on an asset in the middle of a graph.\n\n    now_time --> now_time_plus_N --> now_time_plus_20_after_plus_N\n\n    We are going to iterate on and change the logic of now_time_plus_N in staging and confirm\n    that prod is untouched\n    '
    with DagsterInstance.ephemeral() as prod_instance, DagsterInstance.ephemeral() as dev_instance:
        now_time_plus_N_actually_10 = get_now_time_plus_N(10)
        prod_io_manager = AssetBasedInMemoryIOManager()
        dev_io_manager = AssetBasedInMemoryIOManager()
        prod_runner = DefinitionsRunner(Definitions(assets=[now_time, now_time_plus_N_actually_10, now_time_plus_20_after_plus_N], resources={'io_manager': prod_io_manager}), prod_instance)
        dev_t0_runner = DefinitionsRunner(Definitions(assets=[now_time, now_time_plus_N_actually_10, now_time_plus_20_after_plus_N], resources={'io_manager': BranchingIOManager(parent_io_manager=prod_io_manager, branch_io_manager=dev_io_manager)}), dev_instance)
        assert prod_runner.materialize_all_assets()
        now_time_prod_mat_1 = prod_instance.get_latest_materialization_event(AssetKey('now_time'))
        assert now_time_prod_mat_1
        assert not get_env_entry(now_time_prod_mat_1)
        now_time_prod_value_1 = prod_runner.load_asset_value('now_time')
        assert isinstance(now_time_prod_value_1, int)
        assert dev_t0_runner.materialize_asset('now_time_plus_N').success
        all_mat_event_log_records = dev_t0_runner.get_all_asset_materialization_event_records('now_time_plus_N')
        assert all_mat_event_log_records
        assert len(all_mat_event_log_records) == 1
        now_plus_15_mat_1_log_record = all_mat_event_log_records[0]
        assert get_env_entry(now_plus_15_mat_1_log_record.event_log_entry) == 'dev'
        assert dev_t0_runner.load_asset_value('now_time_plus_N') == now_time_prod_value_1 + 10
        assert not dev_t0_runner.get_all_asset_materialization_event_records('now_time')
        dev_t1_runner = DefinitionsRunner(Definitions(assets=[now_time, get_now_time_plus_N(17), now_time_plus_20_after_plus_N], resources={'io_manager': BranchingIOManager(parent_io_manager=prod_io_manager, branch_io_manager=dev_io_manager)}), dev_instance)
        dev_t1_runner.materialize_asset('now_time_plus_N')
        result = dev_t1_runner.load_asset_value('now_time_plus_N')
        assert result == now_time_prod_value_1 + 17
        assert not dev_t1_runner.get_all_asset_materialization_event_records('now_time')
        dev_t1_runner.materialize_asset('now_time_plus_20_after_plus_N')
        assert dev_t1_runner.load_asset_value('now_time_plus_20_after_plus_N') == now_time_prod_value_1 + 17 + 20