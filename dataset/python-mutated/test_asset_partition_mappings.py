import inspect
from datetime import datetime, timedelta
from typing import Optional
import pytest
from dagster import AllPartitionMapping, AssetExecutionContext, AssetIn, AssetMaterialization, AssetsDefinition, DagsterInvalidDefinitionError, DagsterInvariantViolationError, DailyPartitionsDefinition, IdentityPartitionMapping, IOManager, IOManagerDefinition, LastPartitionMapping, MultiPartitionKey, MultiPartitionsDefinition, MultiToSingleDimensionPartitionMapping, PartitionsDefinition, SourceAsset, SpecificPartitionsPartitionMapping, StaticPartitionsDefinition, TimeWindowPartitionMapping, WeeklyPartitionsDefinition, define_asset_job, graph, instance_for_test, materialize, op
from dagster._core.definitions import asset, build_assets_job
from dagster._core.definitions.asset_dep import AssetDep
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.asset_spec import AssetSpec
from dagster._core.definitions.decorators.asset_decorator import multi_asset
from dagster._core.definitions.events import AssetKey
from dagster._core.definitions.partition import DefaultPartitionsSubset, DynamicPartitionsDefinition, PartitionsSubset
from dagster._core.definitions.partition_key_range import PartitionKeyRange
from dagster._core.definitions.partition_mapping import PartitionMapping, UpstreamPartitionsResult, get_builtin_partition_mapping_types
from dagster._core.instance import DynamicPartitionsStore
from dagster._core.test_utils import assert_namedtuple_lists_equal

def test_access_partition_keys_from_context_non_identity_partition_mapping():
    if False:
        for i in range(10):
            print('nop')
    upstream_partitions_def = StaticPartitionsDefinition(['1', '2', '3'])
    downstream_partitions_def = StaticPartitionsDefinition(['1', '2', '3'])

    class TrailingWindowPartitionMapping(PartitionMapping):
        """Maps each downstream partition to two partitions in the upstream asset: itself and the
        preceding partition.
        """

        def get_upstream_mapped_partitions_result_for_partitions(self, downstream_partitions_subset: Optional[PartitionsSubset], upstream_partitions_def: PartitionsDefinition, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> UpstreamPartitionsResult:
            if False:
                return 10
            assert downstream_partitions_subset
            assert upstream_partitions_def
            partition_keys = list(downstream_partitions_subset.get_partition_keys())
            return UpstreamPartitionsResult(upstream_partitions_def.empty_subset().with_partition_key_range(PartitionKeyRange(str(max(1, int(partition_keys[0]) - 1)), partition_keys[-1])), [])

        def get_downstream_partitions_for_partitions(self, upstream_partitions_subset: PartitionsSubset, downstream_partitions_def: PartitionsDefinition, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> PartitionsSubset:
            if False:
                print('Hello World!')
            raise NotImplementedError()

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            assert context.asset_partition_key == '2'

        def load_input(self, context):
            if False:
                return 10
            (start, end) = context.asset_partition_key_range
            assert start, end == ('1', '2')
            assert context.asset_partitions_def == upstream_partitions_def

    @asset(partitions_def=upstream_partitions_def)
    def upstream_asset(context):
        if False:
            return 10
        assert context.asset_partition_key_for_output() == '2'

    @asset(partitions_def=downstream_partitions_def, ins={'upstream_asset': AssetIn(partition_mapping=TrailingWindowPartitionMapping())})
    def downstream_asset(context, upstream_asset):
        if False:
            while True:
                i = 10
        assert context.asset_partition_key_for_output() == '2'
        assert upstream_asset is None
        assert context.asset_partitions_def_for_input('upstream_asset') == upstream_partitions_def
    my_job = build_assets_job('my_job', assets=[upstream_asset, downstream_asset], resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())})
    result = my_job.execute_in_process(partition_key='2')
    assert_namedtuple_lists_equal(result.asset_materializations_for_node('upstream_asset'), [AssetMaterialization(AssetKey(['upstream_asset']), partition='2')], exclude_fields=['tags'])
    assert_namedtuple_lists_equal(result.asset_materializations_for_node('downstream_asset'), [AssetMaterialization(AssetKey(['downstream_asset']), partition='2')], exclude_fields=['tags'])

def test_from_graph():
    if False:
        for i in range(10):
            print('nop')
    partitions_def = StaticPartitionsDefinition(['a', 'b', 'c', 'd'])

    @op
    def my_op(context):
        if False:
            i = 10
            return i + 15
        assert context.asset_partition_key_for_output() == 'a'

    @graph
    def upstream_asset():
        if False:
            print('Hello World!')
        return my_op()

    @op
    def my_op2(context, upstream_asset):
        if False:
            print('Hello World!')
        assert context.asset_partition_key_for_input('upstream_asset') == 'a'
        assert context.asset_partition_key_for_output() == 'a'
        return upstream_asset

    @graph
    def downstream_asset(upstream_asset):
        if False:
            i = 10
            return i + 15
        return my_op2(upstream_asset)

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                print('Hello World!')
            assert context.asset_partition_key == 'a'
            assert context.has_asset_partitions

        def load_input(self, context):
            if False:
                while True:
                    i = 10
            assert context.asset_partition_key == 'a'
            assert context.has_asset_partitions
    my_job = build_assets_job('my_job', assets=[AssetsDefinition.from_graph(upstream_asset, partitions_def=partitions_def), AssetsDefinition.from_graph(downstream_asset, partitions_def=partitions_def, partition_mappings={'upstream_asset': IdentityPartitionMapping()})], resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())})
    assert my_job.execute_in_process(partition_key='a').success

def test_non_partitioned_depends_on_last_partition():
    if False:
        return 10

    @asset(partitions_def=StaticPartitionsDefinition(['a', 'b', 'c', 'd']))
    def upstream():
        if False:
            for i in range(10):
                print('nop')
        pass

    @asset(ins={'upstream': AssetIn(partition_mapping=LastPartitionMapping())})
    def downstream(upstream):
        if False:
            return 10
        assert upstream is None

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                i = 10
                return i + 15
            if context.asset_key == AssetKey('upstream'):
                assert context.has_asset_partitions
                assert context.asset_partition_key == 'b'
            else:
                assert not context.has_asset_partitions

        def load_input(self, context):
            if False:
                for i in range(10):
                    print('nop')
            assert context.has_asset_partitions
            assert context.asset_partition_key == 'd'
    my_job = build_assets_job('my_job', assets=[upstream, downstream], resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())})
    result = my_job.execute_in_process(partition_key='b')
    assert_namedtuple_lists_equal(result.asset_materializations_for_node('upstream'), [AssetMaterialization(AssetKey(['upstream']), partition='b')], exclude_fields=['tags'])
    assert_namedtuple_lists_equal(result.asset_materializations_for_node('downstream'), [AssetMaterialization(AssetKey(['downstream']))], exclude_fields=['tags'])

def test_non_partitioned_depends_on_specific_partitions():
    if False:
        while True:
            i = 10

    @asset(partitions_def=StaticPartitionsDefinition(['a', 'b', 'c', 'd']))
    def upstream():
        if False:
            return 10
        pass

    @asset(ins={'upstream': AssetIn(partition_mapping=SpecificPartitionsPartitionMapping(['a', 'b']))})
    def downstream_a_b(upstream):
        if False:
            while True:
                i = 10
        assert upstream is None

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                for i in range(10):
                    print('nop')
            if context.asset_key == AssetKey('upstream'):
                assert context.has_asset_partitions
                assert context.asset_partition_key == 'b'
            else:
                assert not context.has_asset_partitions

        def load_input(self, context):
            if False:
                for i in range(10):
                    print('nop')
            assert context.has_asset_partitions
            assert set(context.asset_partition_keys) == {'a', 'b'}
    result = materialize([upstream, downstream_a_b], resources={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())}, partition_key='b')
    assert_namedtuple_lists_equal(result.asset_materializations_for_node('upstream'), [AssetMaterialization(AssetKey(['upstream']), partition='b')], exclude_fields=['tags'])
    assert_namedtuple_lists_equal(result.asset_materializations_for_node('downstream'), [AssetMaterialization(AssetKey(['downstream_a_b']))], exclude_fields=['tags'])

def test_specific_partitions_partition_mapping_downstream_partitions():
    if False:
        while True:
            i = 10
    upstream_partitions_def = StaticPartitionsDefinition(['a', 'b', 'c', 'd'])
    downstream_partitions_def = StaticPartitionsDefinition(['x', 'y', 'z'])
    partition_mapping = SpecificPartitionsPartitionMapping(['a', 'b'])
    for partition_subset in [DefaultPartitionsSubset(upstream_partitions_def, {'a'}), DefaultPartitionsSubset(upstream_partitions_def, {'a', 'b'}), DefaultPartitionsSubset(upstream_partitions_def, {'a', 'b', 'c', 'd'})]:
        assert partition_mapping.get_downstream_partitions_for_partitions(partition_subset, downstream_partitions_def) == downstream_partitions_def.subset_with_all_partitions()
    for partition_subset in [DefaultPartitionsSubset(upstream_partitions_def, {'c'}), DefaultPartitionsSubset(upstream_partitions_def, {'c', 'd'})]:
        assert partition_mapping.get_downstream_partitions_for_partitions(partition_subset, downstream_partitions_def) == downstream_partitions_def.empty_subset()

def test_non_partitioned_depends_on_all_partitions():
    if False:
        i = 10
        return i + 15

    @asset(partitions_def=StaticPartitionsDefinition(['a', 'b', 'c', 'd']))
    def upstream():
        if False:
            while True:
                i = 10
        pass

    @asset(ins={'upstream': AssetIn(partition_mapping=AllPartitionMapping())})
    def downstream(upstream):
        if False:
            print('Hello World!')
        assert upstream is None

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            if context.asset_key == AssetKey('upstream'):
                assert context.has_asset_partitions
                assert context.asset_partition_key == 'b'
            else:
                assert not context.has_asset_partitions

        def load_input(self, context):
            if False:
                print('Hello World!')
            assert context.has_asset_partitions
            assert context.asset_partition_key_range == PartitionKeyRange('a', 'd')
    my_job = build_assets_job('my_job', assets=[upstream, downstream], resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())})
    result = my_job.execute_in_process(partition_key='b')
    assert_namedtuple_lists_equal(result.asset_materializations_for_node('upstream'), [AssetMaterialization(AssetKey(['upstream']), partition='b')], exclude_fields=['tags'])
    assert_namedtuple_lists_equal(result.asset_materializations_for_node('downstream'), [AssetMaterialization(AssetKey(['downstream']))], exclude_fields=['tags'])

def test_partition_keys_in_range():
    if False:
        while True:
            i = 10
    daily_partition_keys_for_week_2022_09_11 = ['2022-09-11', '2022-09-12', '2022-09-13', '2022-09-14', '2022-09-15', '2022-09-16', '2022-09-17']

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2022-09-11'))
    def upstream(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.asset_partition_keys_for_output('result') == ['2022-09-11']
        assert context.asset_partition_keys_for_output() == ['2022-09-11']

    @asset(partitions_def=WeeklyPartitionsDefinition(start_date='2022-09-11'))
    def downstream(context, upstream):
        if False:
            for i in range(10):
                print('nop')
        assert context.asset_partition_keys_for_input('upstream') == daily_partition_keys_for_week_2022_09_11

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                i = 10
                return i + 15
            if context.asset_key == AssetKey('upstream'):
                assert context.has_asset_partitions
                assert context.asset_partition_keys == ['2022-09-11']

        def load_input(self, context):
            if False:
                for i in range(10):
                    print('nop')
            assert context.has_asset_partitions
            assert context.asset_partition_keys == daily_partition_keys_for_week_2022_09_11
    upstream_job = build_assets_job('upstream_job', assets=[upstream], resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())})
    upstream_job.execute_in_process(partition_key='2022-09-11')
    downstream_job = build_assets_job('downstream_job', assets=[downstream], source_assets=[upstream], resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())})
    downstream_job.execute_in_process(partition_key='2022-09-11')

def test_dependency_resolution_partition_mapping():
    if False:
        for i in range(10):
            print('nop')

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2020-01-01'), key_prefix=['staging'])
    def upstream(context):
        if False:
            i = 10
            return i + 15
        partition_date_str = context.asset_partition_key_for_output()
        return partition_date_str

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2020-01-01'), key_prefix=['staging'], ins={'upstream': AssetIn(partition_mapping=TimeWindowPartitionMapping(start_offset=-1, end_offset=0))})
    def downstream(context, upstream):
        if False:
            return 10
        context.log.info(upstream)
        return upstream

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            ...

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            assert context.asset_key.path == ['staging', 'upstream']
            assert context.partition_key == '2020-01-02'
            assert context.asset_partition_keys == ['2020-01-01', '2020-01-02']
    materialize([upstream, downstream], resources={'io_manager': MyIOManager()}, partition_key='2020-01-02')

def test_exported_partition_mappings_whitelisted():
    if False:
        for i in range(10):
            print('nop')
    import dagster
    dagster_exports = (getattr(dagster, attr_name) for attr_name in dagster.__dir__())
    exported_partition_mapping_classes = {export for export in dagster_exports if inspect.isclass(export) and issubclass(export, PartitionMapping)} - {PartitionMapping}
    assert set(get_builtin_partition_mapping_types()) == exported_partition_mapping_classes

def test_multipartitions_def_partition_mapping_infer_identity():
    if False:
        print('Hello World!')
    composite = MultiPartitionsDefinition({'abc': StaticPartitionsDefinition(['a', 'b', 'c']), '123': StaticPartitionsDefinition(['1', '2', '3'])})

    @asset(partitions_def=composite)
    def upstream(context):
        if False:
            while True:
                i = 10
        return 1

    @asset(partitions_def=composite)
    def downstream(context, upstream):
        if False:
            return 10
        assert context.asset_partition_keys_for_input('upstream') == context.asset_partition_keys_for_output()
        return 1
    asset_graph = AssetGraph.from_assets([upstream, downstream])
    assets_job = define_asset_job('foo', [upstream, downstream]).resolve(asset_graph=asset_graph)
    assert asset_graph.get_partition_mapping(upstream.key, downstream.key) == IdentityPartitionMapping()
    assert assets_job.execute_in_process(partition_key=MultiPartitionKey({'abc': 'a', '123': '1'})).success

def test_multipartitions_def_partition_mapping_infer_single_dim_to_multi():
    if False:
        print('Hello World!')
    abc_def = StaticPartitionsDefinition(['a', 'b', 'c'])
    composite = MultiPartitionsDefinition({'abc': abc_def, '123': StaticPartitionsDefinition(['1', '2', '3'])})

    @asset(partitions_def=abc_def)
    def upstream(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.asset_partition_keys_for_output('result') == ['a']
        return 1

    @asset(partitions_def=composite)
    def downstream(context, upstream):
        if False:
            return 10
        assert context.asset_partition_keys_for_input('upstream') == ['a']
        assert context.asset_partition_keys_for_output('result') == [MultiPartitionKey({'abc': 'a', '123': '1'})]
        return 1
    asset_graph = AssetGraph.from_assets([upstream, downstream])
    assert asset_graph.get_partition_mapping(upstream.key, downstream.key) == MultiToSingleDimensionPartitionMapping()

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            if context.asset_key == AssetKey('upstream'):
                assert context.has_asset_partitions
                assert context.asset_partition_keys == ['a']

        def load_input(self, context):
            if False:
                while True:
                    i = 10
            assert context.has_asset_partitions
            assert context.asset_partition_keys == ['a']
    materialize([upstream], resources={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())}, partition_key='a')
    materialize([downstream, SourceAsset(AssetKey('upstream'), partitions_def=abc_def)], resources={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())}, partition_key=MultiPartitionKey({'abc': 'a', '123': '1'}))

def test_multipartitions_def_partition_mapping_infer_multi_to_single_dim():
    if False:
        print('Hello World!')
    abc_def = StaticPartitionsDefinition(['a', 'b', 'c'])
    composite = MultiPartitionsDefinition({'abc': abc_def, '123': StaticPartitionsDefinition(['1', '2', '3'])})
    a_multipartition_keys = {MultiPartitionKey(kv) for kv in [{'abc': 'a', '123': '1'}, {'abc': 'a', '123': '2'}, {'abc': 'a', '123': '3'}]}

    @asset(partitions_def=composite)
    def upstream(context):
        if False:
            i = 10
            return i + 15
        return 1

    @asset(partitions_def=abc_def)
    def downstream(context, upstream):
        if False:
            print('Hello World!')
        assert set(context.asset_partition_keys_for_input('upstream')) == a_multipartition_keys
        assert context.asset_partition_keys_for_output('result') == ['a']
        return 1
    asset_graph = AssetGraph.from_assets([upstream, downstream])
    assert asset_graph.get_partition_mapping(upstream.key, downstream.key) == MultiToSingleDimensionPartitionMapping()

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                i = 10
                return i + 15
            pass

        def load_input(self, context):
            if False:
                print('Hello World!')
            assert set(context.asset_partition_keys) == a_multipartition_keys
    for pk in a_multipartition_keys:
        materialize([upstream], resources={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())}, partition_key=pk)
    materialize([downstream, SourceAsset(AssetKey('upstream'), partitions_def=composite)], resources={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())}, partition_key='a')

def test_identity_partition_mapping():
    if False:
        for i in range(10):
            print('nop')
    xy = StaticPartitionsDefinition(['x', 'y'])
    zx = StaticPartitionsDefinition(['z', 'x'])
    result = IdentityPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(zx.empty_subset().with_partition_keys(['z', 'x']), xy)
    assert result.partitions_subset.get_partition_keys() == set(['x'])
    assert result.required_but_nonexistent_partition_keys == ['z']
    result = IdentityPartitionMapping().get_downstream_partitions_for_partitions(zx.empty_subset().with_partition_keys(['z', 'x']), xy)
    assert result.get_partition_keys() == set(['x'])

def test_partition_mapping_with_asset_deps():
    if False:
        for i in range(10):
            print('nop')
    partitions_def = DailyPartitionsDefinition(start_date='2023-08-15')

    @asset(partitions_def=partitions_def)
    def upstream():
        if False:
            print('Hello World!')
        return

    @asset(partitions_def=partitions_def, deps=[AssetDep(upstream, partition_mapping=TimeWindowPartitionMapping(start_offset=-1, end_offset=-1))])
    def downstream(context: AssetExecutionContext):
        if False:
            i = 10
            return i + 15
        upstream_key = datetime.strptime(context.asset_partition_key_for_input('upstream'), '%Y-%m-%d')
        current_partition_key = datetime.strptime(context.partition_key, '%Y-%m-%d')
        assert current_partition_key - upstream_key == timedelta(days=1)
    materialize([upstream, downstream], partition_key='2023-08-20')
    assert downstream.partition_mappings == {AssetKey('upstream'): TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)}
    asset_1 = AssetSpec(key='asset_1')
    asset_2 = AssetSpec(key='asset_2')
    asset_1_partition_mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)
    asset_2_partition_mapping = TimeWindowPartitionMapping(start_offset=-2, end_offset=-2)
    asset_3 = AssetSpec(key='asset_3', deps=[AssetDep(asset=asset_1, partition_mapping=asset_1_partition_mapping), AssetDep(asset=asset_2, partition_mapping=asset_2_partition_mapping)])
    asset_4 = AssetSpec(key='asset_4', deps=[AssetDep(asset=asset_1, partition_mapping=asset_1_partition_mapping), AssetDep(asset=asset_2, partition_mapping=asset_2_partition_mapping)])

    @multi_asset(specs=[asset_1, asset_2], partitions_def=partitions_def)
    def multi_asset_1():
        if False:
            for i in range(10):
                print('nop')
        return

    @multi_asset(specs=[asset_3, asset_4], partitions_def=partitions_def)
    def multi_asset_2(context: AssetExecutionContext):
        if False:
            for i in range(10):
                print('nop')
        asset_1_key = datetime.strptime(context.asset_partition_key_for_input('asset_1'), '%Y-%m-%d')
        asset_2_key = datetime.strptime(context.asset_partition_key_for_input('asset_2'), '%Y-%m-%d')
        current_partition_key = datetime.strptime(context.partition_key, '%Y-%m-%d')
        assert current_partition_key - asset_1_key == timedelta(days=1)
        assert current_partition_key - asset_2_key == timedelta(days=2)
        return
    materialize([multi_asset_1, multi_asset_2], partition_key='2023-08-20')
    assert multi_asset_2.partition_mappings == {AssetKey('asset_1'): asset_1_partition_mapping, AssetKey('asset_2'): asset_2_partition_mapping}

def test_conflicting_mappings_with_asset_deps():
    if False:
        return 10
    partitions_def = DailyPartitionsDefinition(start_date='2023-08-15')

    @asset(partitions_def=partitions_def)
    def upstream():
        if False:
            i = 10
            return i + 15
        return
    with pytest.raises(DagsterInvariantViolationError, match='Cannot set a dependency on asset'):

        @asset(partitions_def=partitions_def, deps=[AssetDep(upstream, partition_mapping=TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)), AssetDep(upstream, partition_mapping=TimeWindowPartitionMapping(start_offset=-2, end_offset=-2))])
        def downstream():
            if False:
                i = 10
                return i + 15
            pass
    asset_1 = AssetSpec(key='asset_1')
    asset_2 = AssetSpec(key='asset_2')
    asset_1_partition_mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)
    asset_2_partition_mapping = TimeWindowPartitionMapping(start_offset=-2, end_offset=-2)
    asset_3 = AssetSpec(key='asset_3', deps=[AssetDep(asset=asset_1, partition_mapping=asset_1_partition_mapping), AssetDep(asset=asset_2, partition_mapping=asset_2_partition_mapping)])
    asset_4 = AssetSpec(key='asset_4', deps=[AssetDep(asset=asset_1, partition_mapping=asset_1_partition_mapping), AssetDep(asset=asset_2, partition_mapping=TimeWindowPartitionMapping(start_offset=-3, end_offset=-3))])
    with pytest.raises(DagsterInvalidDefinitionError, match='Two different PartitionMappings for'):

        @multi_asset(specs=[asset_3, asset_4], partitions_def=partitions_def)
        def multi_asset_2():
            if False:
                while True:
                    i = 10
            pass

def test_self_dependent_partition_mapping_with_asset_deps():
    if False:
        for i in range(10):
            print('nop')
    partitions_def = DailyPartitionsDefinition(start_date='2023-08-15')

    @asset(partitions_def=partitions_def, deps=[AssetDep('self_dependent', partition_mapping=TimeWindowPartitionMapping(start_offset=-1, end_offset=-1))])
    def self_dependent(context: AssetExecutionContext):
        if False:
            return 10
        upstream_key = datetime.strptime(context.asset_partition_key_for_input('self_dependent'), '%Y-%m-%d')
        current_partition_key = datetime.strptime(context.partition_key, '%Y-%m-%d')
        assert current_partition_key - upstream_key == timedelta(days=1)
    materialize([self_dependent], partition_key='2023-08-20')
    assert self_dependent.partition_mappings == {AssetKey('self_dependent'): TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)}
    asset_1 = AssetSpec(key='asset_1', deps=[AssetDep(asset='asset_1', partition_mapping=TimeWindowPartitionMapping(start_offset=-1, end_offset=-1))])

    @multi_asset(specs=[asset_1], partitions_def=partitions_def)
    def the_multi_asset(context: AssetExecutionContext):
        if False:
            print('Hello World!')
        asset_1_key = datetime.strptime(context.asset_partition_key_for_input('asset_1'), '%Y-%m-%d')
        current_partition_key = datetime.strptime(context.partition_key, '%Y-%m-%d')
        assert current_partition_key - asset_1_key == timedelta(days=1)
    materialize([the_multi_asset], partition_key='2023-08-20')

def test_dynamic_partition_mapping_with_asset_deps():
    if False:
        while True:
            i = 10
    partitions_def = DynamicPartitionsDefinition(name='fruits')

    @asset(partitions_def=partitions_def)
    def upstream():
        if False:
            return 10
        return

    @asset(partitions_def=partitions_def, deps=[AssetDep(upstream, partition_mapping=SpecificPartitionsPartitionMapping(['apple']))])
    def downstream(context: AssetExecutionContext):
        if False:
            print('Hello World!')
        assert context.asset_partition_key_for_input('upstream') == 'apple'
        assert context.partition_key == 'orange'
    with instance_for_test() as instance:
        instance.add_dynamic_partitions('fruits', ['apple'])
        materialize([upstream], partition_key='apple', instance=instance)
        instance.add_dynamic_partitions('fruits', ['orange'])
        materialize([upstream, downstream], partition_key='orange', instance=instance)
    asset_1 = AssetSpec(key='asset_1')
    asset_2 = AssetSpec(key='asset_2', deps=[AssetDep(asset=asset_1, partition_mapping=SpecificPartitionsPartitionMapping(['apple']))])

    @multi_asset(specs=[asset_1], partitions_def=partitions_def)
    def asset_1_multi_asset():
        if False:
            for i in range(10):
                print('nop')
        return

    @multi_asset(specs=[asset_2], partitions_def=partitions_def)
    def asset_2_multi_asset(context: AssetExecutionContext):
        if False:
            for i in range(10):
                print('nop')
        assert context.asset_partition_key_for_input('asset_1') == 'apple'
        assert context.partition_key == 'orange'
    with instance_for_test() as instance:
        instance.add_dynamic_partitions('fruits', ['apple'])
        materialize([asset_1_multi_asset], partition_key='apple', instance=instance)
        instance.add_dynamic_partitions('fruits', ['orange'])
        materialize([asset_1_multi_asset, asset_2_multi_asset], partition_key='orange', instance=instance)

def test_last_partition_mapping_get_downstream_partitions():
    if False:
        while True:
            i = 10
    upstream_partitions_def = DailyPartitionsDefinition('2023-10-01')
    downstream_partitions_def = DailyPartitionsDefinition('2023-10-01')
    current_time = datetime(2023, 10, 5, 1)
    assert LastPartitionMapping().get_downstream_partitions_for_partitions(upstream_partitions_def.empty_subset().with_partition_keys(['2023-10-04']), downstream_partitions_def, current_time) == downstream_partitions_def.empty_subset().with_partition_keys(downstream_partitions_def.get_partition_keys(current_time))
    assert LastPartitionMapping().get_downstream_partitions_for_partitions(upstream_partitions_def.empty_subset().with_partition_keys(['2023-10-03', '2023-10-04']), downstream_partitions_def, current_time) == downstream_partitions_def.empty_subset().with_partition_keys(downstream_partitions_def.get_partition_keys(current_time))
    assert LastPartitionMapping().get_downstream_partitions_for_partitions(upstream_partitions_def.empty_subset().with_partition_keys(['2023-10-03']), downstream_partitions_def, current_time) == downstream_partitions_def.empty_subset()