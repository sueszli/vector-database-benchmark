from typing import Optional
from click.testing import CliRunner
from dagster import AssetKey
from dagster._cli.asset import asset_materialize_command
from dagster._core.test_utils import instance_for_test
from dagster._utils import file_relative_path

def invoke_materialize(select: str, partition: Optional[str]=None):
    if False:
        return 10
    runner = CliRunner()
    options = ['-f', file_relative_path(__file__, 'assets.py'), '--select', select]
    if partition:
        options.extend(['--partition', partition])
    return runner.invoke(asset_materialize_command, options)

def test_empty():
    if False:
        while True:
            i = 10
    with instance_for_test():
        runner = CliRunner()
        result = runner.invoke(asset_materialize_command, [])
        assert result.exit_code == 2
        assert "Missing option '--select'" in result.output

def test_missing_origin():
    if False:
        print('Hello World!')
    with instance_for_test():
        runner = CliRunner()
        result = runner.invoke(asset_materialize_command, ['--select', 'asset1'])
        assert result.exit_code == 2
        assert 'Must specify a python file or module name' in result.output

def test_single_asset():
    if False:
        return 10
    with instance_for_test() as instance:
        result = invoke_materialize('asset1')
        assert 'RUN_SUCCESS' in result.output
        assert instance.get_latest_materialization_event(AssetKey('asset1')) is not None

def test_multi_segment_asset_key():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        result = invoke_materialize('some/key/prefix/asset_with_prefix')
        assert 'RUN_SUCCESS' in result.output
        assert instance.get_latest_materialization_event(AssetKey(['some', 'key', 'prefix', 'asset_with_prefix'])) is not None

def test_asset_with_dep():
    if False:
        return 10
    with instance_for_test() as instance:
        result = invoke_materialize('downstream_asset')
        assert 'RUN_SUCCESS' in result.output
        assert instance.get_latest_materialization_event(AssetKey('downstream_asset')) is not None

def test_two_assets():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test() as instance:
        result = invoke_materialize('asset1,downstream_asset')
        assert 'RUN_SUCCESS' in result.output
        for asset_key in [AssetKey('asset1'), AssetKey('downstream_asset')]:
            assert instance.get_latest_materialization_event(asset_key) is not None

def test_all_downstream():
    if False:
        return 10
    with instance_for_test() as instance:
        result = invoke_materialize('asset1*')
        assert 'RUN_SUCCESS' in result.output
        for asset_key in [AssetKey('asset1'), AssetKey('downstream_asset')]:
            assert instance.get_latest_materialization_event(asset_key) is not None

def test_partition():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        result = invoke_materialize('partitioned_asset', 'one')
        assert 'RUN_SUCCESS' in result.output
        event = instance.get_latest_materialization_event(AssetKey('partitioned_asset'))
        assert event is not None
        assert event.asset_materialization.partition == 'one'

def test_partition_option_with_non_partitioned_asset():
    if False:
        i = 10
        return i + 15
    with instance_for_test():
        result = invoke_materialize('asset1', 'one')
        assert "Provided '--partition' option, but none of the assets are partitioned" in str(result.exception)

def test_asset_key_missing():
    if False:
        print('Hello World!')
    with instance_for_test():
        result = invoke_materialize('nonexistent_asset')
        assert 'No qualified assets to execute found' in str(result.exception)

def test_one_of_the_asset_keys_missing():
    if False:
        return 10
    with instance_for_test():
        result = invoke_materialize('asset1,nonexistent_asset')
        assert 'No qualified assets to execute found' in str(result.exception)

def test_conflicting_partitions():
    if False:
        while True:
            i = 10
    with instance_for_test():
        result = invoke_materialize('partitioned_asset,differently_partitioned_asset')
        assert 'All selected assets must share the same PartitionsDefinition or have no PartitionsDefinition' in str(result.exception)