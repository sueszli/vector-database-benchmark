from typing import AbstractSet, Iterable
import pytest
from dagster import AssetExecutionContext, AssetKey, AssetsDefinition, AutoMaterializePolicy, DagsterInstance, DataVersion, Definitions, IOManager, JobDefinition, SourceAsset, _check as check, asset, observable_source_asset
from dagster._core.definitions.asset_spec import AssetSpec
from dagster._core.definitions.decorators.asset_decorator import multi_asset
from dagster._core.definitions.external_asset import create_external_asset_from_source_asset, external_assets_from_specs
from dagster._core.definitions.freshness_policy import FreshnessPolicy
from dagster._core.definitions.time_window_partitions import DailyPartitionsDefinition

def test_external_asset_basic_creation() -> None:
    if False:
        i = 10
        return i + 15
    assets_def = next(iter(external_assets_from_specs(specs=[AssetSpec(key='external_asset_one', description='desc', metadata={'user_metadata': 'value'}, group_name='a_group')])))
    assert isinstance(assets_def, AssetsDefinition)
    expected_key = AssetKey(['external_asset_one'])
    assert assets_def.key == expected_key
    assert assets_def.metadata_by_key[expected_key]['user_metadata'] == 'value'
    assert assets_def.group_names_by_key[expected_key] == 'a_group'
    assert assets_def.descriptions_by_key[expected_key] == 'desc'
    assert assets_def.is_asset_executable(expected_key) is False

def test_multi_external_asset_basic_creation() -> None:
    if False:
        for i in range(10):
            print('nop')
    for assets_def in external_assets_from_specs(specs=[AssetSpec(key='external_asset_one', description='desc', metadata={'user_metadata': 'value'}, group_name='a_group'), AssetSpec(key=AssetKey(['value', 'another_spec']), description='desc', metadata={'user_metadata': 'value'}, group_name='a_group')]):
        assert isinstance(assets_def, AssetsDefinition)

def test_invalid_external_asset_creation() -> None:
    if False:
        return 10
    invalid_specs = [AssetSpec('invalid_asset1', auto_materialize_policy=AutoMaterializePolicy.eager()), AssetSpec('invalid_asset2', code_version='ksjdfljs'), AssetSpec('invalid_asset2', freshness_policy=FreshnessPolicy(maximum_lag_minutes=1)), AssetSpec('invalid_asset2', skippable=True)]
    for invalid_spec in invalid_specs:
        with pytest.raises(check.CheckError):
            external_assets_from_specs(specs=[invalid_spec])

def test_normal_asset_materializeable() -> None:
    if False:
        for i in range(10):
            print('nop')

    @asset
    def an_asset() -> None:
        if False:
            i = 10
            return i + 15
        ...
    assert an_asset.is_asset_executable(AssetKey(['an_asset'])) is True

def test_external_asset_creation_with_deps() -> None:
    if False:
        for i in range(10):
            print('nop')
    asset_two = AssetSpec('external_asset_two')
    assets_def = next(iter(external_assets_from_specs([AssetSpec('external_asset_one', deps=[asset_two.key])])))
    assert isinstance(assets_def, AssetsDefinition)
    expected_key = AssetKey(['external_asset_one'])
    assert assets_def.key == expected_key
    assert assets_def.asset_deps[expected_key] == {AssetKey(['external_asset_two'])}

def test_how_source_assets_are_backwards_compatible() -> None:
    if False:
        for i in range(10):
            print('nop')

    class DummyIOManager(IOManager):

        def handle_output(self, context, obj) -> None:
            if False:
                i = 10
                return i + 15
            pass

        def load_input(self, context) -> str:
            if False:
                print('Hello World!')
            return 'hardcoded'
    source_asset = SourceAsset(key='source_asset', io_manager_def=DummyIOManager())

    @asset
    def an_asset(source_asset: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return source_asset + '-computed'
    defs_with_source = Definitions(assets=[source_asset, an_asset])
    instance = DagsterInstance.ephemeral()
    result_one = defs_with_source.get_implicit_global_asset_job_def().execute_in_process(instance=instance)
    assert result_one.success
    assert result_one.output_for_node('an_asset') == 'hardcoded-computed'
    defs_with_shim = Definitions(assets=[create_external_asset_from_source_asset(source_asset), an_asset])
    assert isinstance(defs_with_shim.get_assets_def('source_asset'), AssetsDefinition)
    result_two = defs_with_shim.get_implicit_global_asset_job_def().execute_in_process(instance=instance, asset_selection=[AssetKey('an_asset')])
    assert result_two.success
    assert result_two.output_for_node('an_asset') == 'hardcoded-computed'

def get_job_for_assets(defs: Definitions, *coercibles_or_defs) -> JobDefinition:
    if False:
        return 10
    job_def = defs.get_implicit_job_def_for_assets(set_from_coercibles_or_defs(coercibles_or_defs))
    assert job_def, 'Expected to find a job def'
    return job_def

def set_from_coercibles_or_defs(coercibles_or_defs: Iterable) -> AbstractSet['AssetKey']:
    if False:
        while True:
            i = 10
    return set([AssetKey.from_coercible_or_definition(coercible_or_def) for coercible_or_def in coercibles_or_defs])

def test_how_partitioned_source_assets_are_backwards_compatible() -> None:
    if False:
        print('Hello World!')

    class DummyIOManager(IOManager):

        def handle_output(self, context, obj) -> None:
            if False:
                print('Hello World!')
            pass

        def load_input(self, context) -> str:
            if False:
                i = 10
                return i + 15
            return 'hardcoded'
    partitions_def = DailyPartitionsDefinition(start_date='2021-01-01')
    source_asset = SourceAsset(key='source_asset', io_manager_def=DummyIOManager(), partitions_def=partitions_def)

    @asset(partitions_def=partitions_def)
    def an_asset(context: AssetExecutionContext, source_asset: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return source_asset + '-computed-' + context.partition_key
    assert an_asset.partitions_def is partitions_def
    assert source_asset.partitions_def is partitions_def
    defs_with_source = Definitions(assets=[source_asset, an_asset])
    instance = DagsterInstance.ephemeral()
    job_def_without_shim = get_job_for_assets(defs_with_source, an_asset)
    result_one = job_def_without_shim.execute_in_process(instance=instance, partition_key='2021-01-02')
    assert result_one.success
    assert result_one.output_for_node('an_asset') == 'hardcoded-computed-2021-01-02'
    shimmed_source_asset = create_external_asset_from_source_asset(source_asset)
    defs_with_shim = Definitions(assets=[create_external_asset_from_source_asset(source_asset), an_asset])
    assert isinstance(defs_with_shim.get_assets_def('source_asset'), AssetsDefinition)
    job_def_with_shim = get_job_for_assets(defs_with_shim, an_asset, shimmed_source_asset)
    result_two = job_def_with_shim.execute_in_process(instance=instance, asset_selection=[AssetKey('an_asset')], partition_key='2021-01-03')
    assert result_two.success
    assert result_two.output_for_node('an_asset') == 'hardcoded-computed-2021-01-03'

def test_observable_source_asset_decorator() -> None:
    if False:
        while True:
            i = 10

    @observable_source_asset
    def an_observable_source_asset() -> DataVersion:
        if False:
            return 10
        return DataVersion('foo')
    assets_def = create_external_asset_from_source_asset(an_observable_source_asset)
    assert assets_def.is_asset_executable(an_observable_source_asset.key)
    defs = Definitions(assets=[assets_def])
    instance = DagsterInstance.ephemeral()
    result = defs.get_implicit_global_asset_job_def().execute_in_process(instance=instance)
    assert result.success
    assert result.output_for_node('an_observable_source_asset') is None
    all_observations = result.get_asset_observation_events()
    assert len(all_observations) == 1
    observation_event = all_observations[0]
    assert observation_event.asset_observation_data.asset_observation.data_version == 'foo'
    all_materializations = result.get_asset_materialization_events()
    assert len(all_materializations) == 0

def test_external_assets_with_dependencies_manual_construction() -> None:
    if False:
        return 10
    upstream_asset = AssetSpec('upstream_asset')
    downstream_asset = AssetSpec('downstream_asset', deps=[upstream_asset])

    @multi_asset(name='_generated_asset_def_1', specs=[upstream_asset])
    def _upstream_def(context: AssetExecutionContext) -> None:
        if False:
            i = 10
            return i + 15
        raise Exception('do not execute')

    @multi_asset(name='_generated_asset_def_2', specs=[downstream_asset])
    def _downstream_asset(context: AssetExecutionContext) -> None:
        if False:
            return 10
        raise Exception('do not execute')
    defs = Definitions(assets=[_upstream_def, _downstream_asset])
    assert defs
    assert defs.get_implicit_global_asset_job_def().asset_layer.asset_deps[AssetKey('downstream_asset')] == {AssetKey('upstream_asset')}

def test_external_asset_multi_asset() -> None:
    if False:
        i = 10
        return i + 15
    upstream_asset = AssetSpec('upstream_asset')
    downstream_asset = AssetSpec('downstream_asset', deps=[upstream_asset])

    @multi_asset(specs=[downstream_asset, upstream_asset])
    def _generated_asset_def(context: AssetExecutionContext):
        if False:
            while True:
                i = 10
        raise Exception('do not execute')
    defs = Definitions(assets=[_generated_asset_def])
    assert defs
    assert defs.get_implicit_global_asset_job_def().asset_layer.asset_deps[AssetKey('downstream_asset')] == {AssetKey('upstream_asset')}

def test_external_assets_with_dependencies() -> None:
    if False:
        return 10
    upstream_asset = AssetSpec('upstream_asset')
    downstream_asset = AssetSpec('downstream_asset', deps=[upstream_asset])
    defs = Definitions(assets=external_assets_from_specs([upstream_asset, downstream_asset]))
    assert defs
    assert defs.get_implicit_global_asset_job_def().asset_layer.asset_deps[AssetKey('downstream_asset')] == {AssetKey('upstream_asset')}