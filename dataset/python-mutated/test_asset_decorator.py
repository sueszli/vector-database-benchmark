import json
import os
from pathlib import Path
from typing import AbstractSet, Any, Mapping, Optional
import pytest
from dagster import AssetKey, AutoMaterializePolicy, BackfillPolicy, DagsterInvalidDefinitionError, DailyPartitionsDefinition, Definitions, DependencyDefinition, FreshnessPolicy, LastPartitionMapping, NodeInvocation, PartitionMapping, PartitionsDefinition, TimeWindowPartitionMapping, asset
from dagster._core.definitions.utils import DEFAULT_IO_MANAGER_KEY
from dagster._core.execution.context.compute import AssetExecutionContext
from dagster_dbt.asset_decorator import dbt_assets
from dagster_dbt.core.resources_v2 import DbtCliResource
from dagster_dbt.dagster_dbt_translator import DagsterDbtTranslator
from dagster_dbt.dbt_manifest import DbtManifestParam
pytest.importorskip('dbt.version', minversion='1.4')
manifest_path = Path(__file__).joinpath('..', 'sample_manifest.json').resolve()
manifest = json.loads(manifest_path.read_bytes())
test_dagster_metadata_manifest_path = Path(__file__).joinpath('..', 'dbt_projects', 'test_dagster_metadata', 'manifest.json').resolve()
test_dagster_metadata_manifest = json.loads(test_dagster_metadata_manifest_path.read_bytes())
test_python_interleaving_manifest_path = Path(__file__).joinpath('..', 'dbt_projects', 'test_dagster_dbt_python_interleaving', 'manifest.json').resolve()
test_python_interleaving_manifest = json.loads(test_python_interleaving_manifest_path.read_bytes())

@pytest.mark.parametrize('manifest', [manifest, manifest_path, os.fspath(manifest_path)])
def test_manifest_argument(manifest: DbtManifestParam):
    if False:
        print('Hello World!')

    @dbt_assets(manifest=manifest)
    def my_dbt_assets():
        if False:
            i = 10
            return i + 15
        ...
    assert my_dbt_assets.keys == {AssetKey.from_user_string(key) for key in ['sort_by_calories', 'cold_schema/sort_cold_cereals_by_calories', 'subdir_schema/least_caloric', 'sort_hot_cereals_by_calories', 'orders_snapshot', 'cereals']}

@pytest.mark.parametrize('select,exclude,expected_asset_names', [('*', None, {'sort_by_calories', 'cold_schema/sort_cold_cereals_by_calories', 'subdir_schema/least_caloric', 'sort_hot_cereals_by_calories', 'orders_snapshot', 'cereals'}), ('+least_caloric', None, {'sort_by_calories', 'subdir_schema/least_caloric', 'cereals'}), ('sort_by_calories least_caloric', None, {'sort_by_calories', 'subdir_schema/least_caloric'}), ('tag:bar+', None, {'sort_by_calories', 'cold_schema/sort_cold_cereals_by_calories', 'subdir_schema/least_caloric', 'sort_hot_cereals_by_calories', 'orders_snapshot'}), ('tag:foo', None, {'sort_by_calories', 'cold_schema/sort_cold_cereals_by_calories'}), ('tag:foo,tag:bar', None, {'sort_by_calories'}), (None, 'sort_hot_cereals_by_calories', {'sort_by_calories', 'cold_schema/sort_cold_cereals_by_calories', 'subdir_schema/least_caloric', 'cereals', 'orders_snapshot'}), (None, '+least_caloric', {'cold_schema/sort_cold_cereals_by_calories', 'sort_hot_cereals_by_calories', 'orders_snapshot'}), (None, 'sort_by_calories least_caloric', {'cold_schema/sort_cold_cereals_by_calories', 'sort_hot_cereals_by_calories', 'orders_snapshot', 'cereals'}), (None, 'tag:foo', {'subdir_schema/least_caloric', 'sort_hot_cereals_by_calories', 'orders_snapshot', 'cereals'}), ('*', 'tag:does-not-exist', {'sort_by_calories', 'cold_schema/sort_cold_cereals_by_calories', 'subdir_schema/least_caloric', 'sort_hot_cereals_by_calories', 'orders_snapshot', 'cereals'})])
def test_selections(select: Optional[str], exclude: Optional[str], expected_asset_names: AbstractSet[str]) -> None:
    if False:
        i = 10
        return i + 15

    @dbt_assets(manifest=manifest, select=select or 'fqn:*', exclude=exclude)
    def my_dbt_assets():
        if False:
            i = 10
            return i + 15
        ...
    expected_asset_keys = {AssetKey(key.split('/')) for key in expected_asset_names}
    assert my_dbt_assets.keys == expected_asset_keys
    expected_select_tag = 'fqn:*' if select is None else select
    assert my_dbt_assets.op.tags.get('dagster-dbt/select') == expected_select_tag
    assert my_dbt_assets.op.tags.get('dagster-dbt/exclude') == exclude

@pytest.mark.parametrize('name', [None, 'custom'])
def test_with_custom_name(name: Optional[str]) -> None:
    if False:
        print('Hello World!')

    @dbt_assets(manifest=manifest, name=name)
    def my_dbt_assets():
        if False:
            for i in range(10):
                print('nop')
        ...
    expected_name = name or 'my_dbt_assets'
    assert my_dbt_assets.op.name == expected_name

@pytest.mark.parametrize('partitions_def', [None, DailyPartitionsDefinition(start_date='2023-01-01')])
def test_partitions_def(partitions_def: Optional[PartitionsDefinition]) -> None:
    if False:
        return 10

    @dbt_assets(manifest=manifest, partitions_def=partitions_def)
    def my_dbt_assets():
        if False:
            while True:
                i = 10
        ...
    assert my_dbt_assets.partitions_def == partitions_def

@pytest.mark.parametrize('io_manager_key', [None, 'my_io_manager_key'])
def test_io_manager_key(io_manager_key: Optional[str]) -> None:
    if False:
        i = 10
        return i + 15

    @dbt_assets(manifest=manifest, io_manager_key=io_manager_key)
    def my_dbt_assets():
        if False:
            while True:
                i = 10
        ...
    expected_io_manager_key = DEFAULT_IO_MANAGER_KEY if io_manager_key is None else io_manager_key
    for output_def in my_dbt_assets.node_def.output_defs:
        assert output_def.io_manager_key == expected_io_manager_key

def test_backfill_policy():
    if False:
        i = 10
        return i + 15
    backfill_policy = BackfillPolicy.single_run()

    @dbt_assets(manifest=manifest, partitions_def=DailyPartitionsDefinition(start_date='2023-01-01'), backfill_policy=backfill_policy)
    def my_dbt_assets():
        if False:
            print('Hello World!')
        ...
    assert my_dbt_assets.backfill_policy == backfill_policy

def test_op_tags():
    if False:
        i = 10
        return i + 15

    @dbt_assets(manifest=manifest, op_tags={'a': 'b', 'c': 'd'})
    def my_dbt_assets():
        if False:
            for i in range(10):
                print('nop')
        ...
    assert my_dbt_assets.op.tags == {'a': 'b', 'c': 'd', 'kind': 'dbt', 'dagster-dbt/select': 'fqn:*'}

    @dbt_assets(manifest=manifest, op_tags={'a': 'b', 'c': 'd'}, select='+least_caloric')
    def my_dbt_assets_with_select():
        if False:
            i = 10
            return i + 15
        ...
    assert my_dbt_assets_with_select.op.tags == {'a': 'b', 'c': 'd', 'kind': 'dbt', 'dagster-dbt/select': '+least_caloric'}

    @dbt_assets(manifest=manifest, op_tags={'a': 'b', 'c': 'd'}, exclude='+least_caloric')
    def my_dbt_assets_with_exclude():
        if False:
            for i in range(10):
                print('nop')
        ...
    assert my_dbt_assets_with_exclude.op.tags == {'a': 'b', 'c': 'd', 'kind': 'dbt', 'dagster-dbt/exclude': '+least_caloric', 'dagster-dbt/select': 'fqn:*'}

    @dbt_assets(manifest=manifest, op_tags={'a': 'b', 'c': 'd'}, select='+least_caloric', exclude='least_caloric')
    def my_dbt_assets_with_select_and_exclude():
        if False:
            while True:
                i = 10
        ...
    assert my_dbt_assets_with_select_and_exclude.op.tags == {'a': 'b', 'c': 'd', 'kind': 'dbt', 'dagster-dbt/select': '+least_caloric', 'dagster-dbt/exclude': 'least_caloric'}
    with pytest.raises(DagsterInvalidDefinitionError, match="To specify a dbt selection, use the 'select' argument, not 'dagster-dbt/select' with op_tags"):

        @dbt_assets(manifest=manifest, op_tags={'a': 'b', 'c': 'd', 'dagster-dbt/select': '+least_caloric'})
        def select_tag():
            if False:
                for i in range(10):
                    print('nop')
            ...
    with pytest.raises(DagsterInvalidDefinitionError, match="To specify a dbt exclusion, use the 'exclude' argument, not 'dagster-dbt/exclude' with op_tags"):

        @dbt_assets(manifest=manifest, op_tags={'a': 'b', 'c': 'd', 'dagster-dbt/exclude': '+least_caloric'})
        def exclude_tag():
            if False:
                print('Hello World!')
            ...

def test_with_asset_key_replacements() -> None:
    if False:
        i = 10
        return i + 15

    class CustomizedDagsterDbtTranslator(DagsterDbtTranslator):

        @classmethod
        def get_asset_key(cls, dbt_resource_props: Mapping[str, Any]) -> AssetKey:
            if False:
                while True:
                    i = 10
            return AssetKey(['prefix', *super().get_asset_key(dbt_resource_props).path])

    @dbt_assets(manifest=manifest, dagster_dbt_translator=CustomizedDagsterDbtTranslator())
    def my_dbt_assets():
        if False:
            return 10
        ...
    assert my_dbt_assets.keys_by_input_name == {'__subset_input__cereals': AssetKey(['prefix', 'cereals']), '__subset_input__sort_by_calories': AssetKey(['prefix', 'sort_by_calories'])}
    assert set(my_dbt_assets.keys_by_output_name.values()) == {AssetKey(['prefix', 'cereals']), AssetKey(['prefix', 'cold_schema', 'sort_cold_cereals_by_calories']), AssetKey(['prefix', 'subdir_schema', 'least_caloric']), AssetKey(['prefix', 'orders_snapshot']), AssetKey(['prefix', 'sort_hot_cereals_by_calories']), AssetKey(['prefix', 'sort_by_calories'])}

@pytest.mark.parametrize('partition_mapping', [None, LastPartitionMapping(), TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)])
def test_with_partition_mappings(partition_mapping: Optional[PartitionMapping]) -> None:
    if False:
        print('Hello World!')
    expected_self_dependency_partition_mapping = TimeWindowPartitionMapping(start_offset=-8, end_offset=-9)

    class CustomizedDagsterDbtTranslator(DagsterDbtTranslator):

        @classmethod
        def get_partition_mapping(cls, dbt_resource_props: Mapping[str, Any], dbt_parent_resource_props: Mapping[str, Any]) -> Optional[PartitionMapping]:
            if False:
                print('Hello World!')
            is_self_dependency = dbt_resource_props == dbt_parent_resource_props
            if is_self_dependency:
                return expected_self_dependency_partition_mapping
            return partition_mapping

    @dbt_assets(manifest=test_dagster_metadata_manifest, dagster_dbt_translator=CustomizedDagsterDbtTranslator(), partitions_def=DailyPartitionsDefinition(start_date='2023-10-01'))
    def my_dbt_assets():
        if False:
            return 10
        ...
    dependencies_with_self_dependencies = {AssetKey('customers')}
    dependencies_without_self_dependencies = set(my_dbt_assets.dependency_keys).difference(my_dbt_assets.keys)
    assert dependencies_without_self_dependencies
    for input_asset_key in dependencies_without_self_dependencies:
        assert my_dbt_assets.get_partition_mapping(input_asset_key) == partition_mapping
    for self_dependency_asset_key in dependencies_with_self_dependencies:
        assert my_dbt_assets.get_partition_mapping(self_dependency_asset_key) == expected_self_dependency_partition_mapping

def test_with_description_replacements() -> None:
    if False:
        print('Hello World!')
    expected_description = 'customized description'

    class CustomizedDagsterDbtTranslator(DagsterDbtTranslator):

        @classmethod
        def get_description(cls, dbt_resource_props: Mapping[str, Any]) -> str:
            if False:
                i = 10
                return i + 15
            return expected_description

    @dbt_assets(manifest=manifest, dagster_dbt_translator=CustomizedDagsterDbtTranslator())
    def my_dbt_assets():
        if False:
            return 10
        ...
    for description in my_dbt_assets.descriptions_by_key.values():
        assert description == expected_description

def test_with_metadata_replacements() -> None:
    if False:
        print('Hello World!')
    expected_metadata = {'customized': 'metadata'}

    class CustomizedDagsterDbtTranslator(DagsterDbtTranslator):

        @classmethod
        def get_metadata(cls, dbt_resource_props: Mapping[str, Any]) -> Mapping[str, Any]:
            if False:
                i = 10
                return i + 15
            return expected_metadata

    @dbt_assets(manifest=manifest, dagster_dbt_translator=CustomizedDagsterDbtTranslator())
    def my_dbt_assets():
        if False:
            return 10
        ...
    for metadata in my_dbt_assets.metadata_by_key.values():
        assert metadata['customized'] == 'metadata'

def test_with_group_replacements() -> None:
    if False:
        for i in range(10):
            print('nop')
    expected_group = 'customized_group'

    class CustomizedDagsterDbtTranslator(DagsterDbtTranslator):

        @classmethod
        def get_group_name(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[str]:
            if False:
                for i in range(10):
                    print('nop')
            return expected_group

    @dbt_assets(manifest=manifest, dagster_dbt_translator=CustomizedDagsterDbtTranslator())
    def my_dbt_assets():
        if False:
            for i in range(10):
                print('nop')
        ...
    for group in my_dbt_assets.group_names_by_key.values():
        assert group == expected_group

def test_with_freshness_policy_replacements() -> None:
    if False:
        i = 10
        return i + 15
    expected_freshness_policy = FreshnessPolicy(maximum_lag_minutes=60)

    class CustomizedDagsterDbtTranslator(DagsterDbtTranslator):

        @classmethod
        def get_freshness_policy(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[FreshnessPolicy]:
            if False:
                print('Hello World!')
            return expected_freshness_policy

    @dbt_assets(manifest=manifest, dagster_dbt_translator=CustomizedDagsterDbtTranslator())
    def my_dbt_assets():
        if False:
            print('Hello World!')
        ...
    for freshness_policy in my_dbt_assets.freshness_policies_by_key.values():
        assert freshness_policy == expected_freshness_policy

def test_with_auto_materialize_policy_replacements() -> None:
    if False:
        print('Hello World!')
    expected_auto_materialize_policy = AutoMaterializePolicy.eager()

    class CustomizedDagsterDbtTranslator(DagsterDbtTranslator):

        @classmethod
        def get_auto_materialize_policy(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[AutoMaterializePolicy]:
            if False:
                for i in range(10):
                    print('nop')
            return expected_auto_materialize_policy

    @dbt_assets(manifest=manifest, dagster_dbt_translator=CustomizedDagsterDbtTranslator())
    def my_dbt_assets():
        if False:
            i = 10
            return i + 15
        ...
    for auto_materialize_policy in my_dbt_assets.auto_materialize_policies_by_key.values():
        assert auto_materialize_policy == expected_auto_materialize_policy

def test_dbt_meta_auto_materialize_policy() -> None:
    if False:
        print('Hello World!')

    @dbt_assets(manifest=test_dagster_metadata_manifest)
    def my_dbt_assets():
        if False:
            print('Hello World!')
        ...
    auto_materialize_policies = my_dbt_assets.auto_materialize_policies_by_key.values()
    assert auto_materialize_policies
    for auto_materialize_policy in auto_materialize_policies:
        assert auto_materialize_policy == AutoMaterializePolicy.eager()

def test_dbt_meta_freshness_policy() -> None:
    if False:
        while True:
            i = 10

    @dbt_assets(manifest=test_dagster_metadata_manifest)
    def my_dbt_assets():
        if False:
            i = 10
            return i + 15
        ...
    freshness_policies = my_dbt_assets.freshness_policies_by_key.values()
    assert freshness_policies
    for freshness_policy in freshness_policies:
        assert freshness_policy == FreshnessPolicy(maximum_lag_minutes=60.0, cron_schedule='* * * * *')

def test_dbt_meta_asset_key() -> None:
    if False:
        while True:
            i = 10

    @dbt_assets(manifest=test_dagster_metadata_manifest)
    def my_dbt_assets():
        if False:
            for i in range(10):
                print('nop')
        ...
    assert AssetKey(['customized', 'source', 'jaffle_shop', 'main', 'raw_customers']) in set(my_dbt_assets.keys_by_input_name.values())
    assert {AssetKey(['customized', 'staging', 'customers']), AssetKey(['customized', 'staging', 'orders']), AssetKey(['customized', 'staging', 'payments'])}.issubset(my_dbt_assets.keys)

def test_dbt_config_group() -> None:
    if False:
        return 10

    @dbt_assets(manifest=test_dagster_metadata_manifest)
    def my_dbt_assets():
        if False:
            for i in range(10):
                print('nop')
        ...
    assert my_dbt_assets.group_names_by_key == {AssetKey(['customers']): 'default', AssetKey(['customized', 'staging', 'customers']): 'customized_dagster_group', AssetKey(['customized', 'staging', 'orders']): 'customized_dbt_group', AssetKey(['customized', 'staging', 'payments']): 'customized_dagster_group', AssetKey(['orders']): 'default', AssetKey(['raw_customers']): 'default', AssetKey(['raw_orders']): 'default', AssetKey(['raw_payments']): 'default'}

def test_dbt_with_downstream_asset_via_definition():
    if False:
        for i in range(10):
            print('nop')

    @dbt_assets(manifest=test_dagster_metadata_manifest)
    def my_dbt_assets():
        if False:
            while True:
                i = 10
        ...

    @asset(deps=[my_dbt_assets])
    def downstream_of_dbt():
        if False:
            for i in range(10):
                print('nop')
        return None
    assert len(downstream_of_dbt.input_names) == 8
    for input_name in downstream_of_dbt.input_names:
        assert downstream_of_dbt.op.ins[input_name].dagster_type.is_nothing

def test_dbt_with_downstream_asset():
    if False:
        for i in range(10):
            print('nop')

    @dbt_assets(manifest=test_dagster_metadata_manifest)
    def my_dbt_assets():
        if False:
            return 10
        ...

    @asset(deps=[AssetKey('orders'), AssetKey(['customized', 'staging', 'payments'])])
    def downstream_of_dbt():
        if False:
            print('Hello World!')
        return None
    assert len(downstream_of_dbt.input_names) == 2
    assert downstream_of_dbt.op.ins['orders'].dagster_type.is_nothing
    assert downstream_of_dbt.op.ins['customized_staging_payments'].dagster_type.is_nothing

def test_dbt_with_python_interleaving() -> None:
    if False:
        return 10

    @dbt_assets(manifest=test_python_interleaving_manifest)
    def my_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
        if False:
            return 10
        yield from dbt.cli(['build'], context=context).stream()
    assert set(my_dbt_assets.keys_by_input_name.values()) == {AssetKey(['dagster', 'python_augmented_customers']), AssetKey('raw_customers'), AssetKey('raw_orders'), AssetKey('raw_payments'), AssetKey('stg_orders'), AssetKey('stg_payments')}

    @asset(key_prefix='dagster', deps=['raw_customers'])
    def python_augmented_customers():
        if False:
            i = 10
            return i + 15
        ...
    defs = Definitions(assets=[my_dbt_assets, python_augmented_customers], resources={'dbt': DbtCliResource(project_dir=test_python_interleaving_manifest_path.parent.absolute().as_posix())})
    global_job = defs.get_implicit_global_asset_job_def()
    assert global_job.dependencies == {NodeInvocation(name='my_dbt_assets', alias='my_dbt_assets_2'): {}, NodeInvocation(name='dagster__python_augmented_customers'): {'raw_customers': DependencyDefinition(node='my_dbt_assets_2', output='raw_customers')}, NodeInvocation(name='my_dbt_assets'): {'__subset_input__stg_orders': DependencyDefinition(node='my_dbt_assets_2', output='stg_orders'), '__subset_input__stg_payments': DependencyDefinition(node='my_dbt_assets_2', output='stg_payments'), 'dagster_python_augmented_customers': DependencyDefinition(node='dagster__python_augmented_customers', output='result')}}
    assert len(global_job.all_node_defs) == 2
    assert len(global_job.nodes) == 3
    result = global_job.execute_in_process()
    assert result.success
    subset_job = global_job.get_subset(asset_selection={AssetKey('stg_orders'), AssetKey('customers')})
    assert subset_job.dependencies == {NodeInvocation(name='my_dbt_assets', alias='my_dbt_assets_2'): {}, NodeInvocation(name='my_dbt_assets'): {'__subset_input__stg_orders': DependencyDefinition(node='my_dbt_assets_2', output='stg_orders')}}
    result = subset_job.execute_in_process()
    assert result.success