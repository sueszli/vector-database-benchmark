import pytest
from dagster import AssetKey, AssetOut, FilesystemIOManager, IOManager, SourceAsset, TimeWindowPartitionMapping, asset, materialize, multi_asset
from dagster._check import ParameterCheckError
from dagster._core.definitions.asset_dep import AssetDep
from dagster._core.definitions.asset_spec import AssetSpec
from dagster._core.errors import DagsterInvalidDefinitionError, DagsterInvariantViolationError
from dagster._core.types.dagster_type import DagsterTypeKind

def test_basic_instantiation():
    if False:
        i = 10
        return i + 15

    @asset
    def upstream():
        if False:
            while True:
                i = 10
        pass
    assert AssetDep('upstream').asset_key == upstream.key
    assert AssetDep(upstream).asset_key == upstream.key
    assert AssetDep(AssetKey(['upstream'])).asset_key == upstream.key
    partition_mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)
    assert AssetDep('upstream', partition_mapping=partition_mapping).partition_mapping == partition_mapping
    the_source = SourceAsset(key='the_source')
    assert AssetDep(the_source).asset_key == the_source.key

def test_instantiation_with_asset_dep():
    if False:
        for i in range(10):
            print('nop')
    partition_mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)
    og_dep = AssetDep('upstream', partition_mapping=partition_mapping)
    with pytest.raises(ParameterCheckError):
        assert AssetDep(og_dep) == AssetDep('upstream')

def test_multi_asset_errors():
    if False:
        return 10

    @multi_asset(specs=[AssetSpec('asset_1'), AssetSpec('asset_2')])
    def a_multi_asset():
        if False:
            return 10
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match='Cannot create an AssetDep from a multi_asset AssetsDefinition'):
        AssetDep(a_multi_asset)

def test_from_coercible():
    if False:
        print('Hello World!')
    compare_dep = AssetDep('upstream')

    @asset
    def upstream():
        if False:
            for i in range(10):
                print('nop')
        pass
    assert AssetDep.from_coercible(upstream) == compare_dep
    assert AssetDep.from_coercible('upstream') == compare_dep
    assert AssetDep.from_coercible(AssetKey(['upstream'])) == compare_dep
    assert AssetDep.from_coercible(compare_dep) == compare_dep
    the_source = SourceAsset(key='the_source')
    source_compare_dep = AssetDep(the_source)
    assert AssetDep.from_coercible(the_source) == source_compare_dep
    partition_mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)
    with_partition_mapping = AssetDep('with_partition_mapping', partition_mapping=partition_mapping)
    assert AssetDep.from_coercible(with_partition_mapping) == with_partition_mapping

    @multi_asset(specs=[AssetSpec('asset_1'), AssetSpec('asset_2')])
    def a_multi_asset():
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match='Cannot create an AssetDep from a multi_asset AssetsDefinition'):
        AssetDep.from_coercible(a_multi_asset)
    with pytest.raises(ParameterCheckError, match='Param "asset" is not one of'):
        AssetDep.from_coercible(1)

class TestingIOManager(IOManager):

    def handle_output(self, context, obj):
        if False:
            for i in range(10):
                print('nop')
        return None

    def load_input(self, context):
        if False:
            i = 10
            return i + 15
        assert False

def test_single_asset_deps_via_asset_dep():
    if False:
        print('Hello World!')

    @asset
    def asset_1():
        if False:
            i = 10
            return i + 15
        return None

    @asset(deps=[AssetDep(asset_1)])
    def asset_2():
        if False:
            while True:
                i = 10
        return None
    assert len(asset_2.input_names) == 1
    assert asset_2.op.ins['asset_1'].dagster_type.is_nothing
    res = materialize([asset_1, asset_2], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_single_asset_deps_via_assets_definition():
    if False:
        i = 10
        return i + 15

    @asset
    def asset_1():
        if False:
            for i in range(10):
                print('nop')
        return None

    @asset(deps=[asset_1])
    def asset_2():
        if False:
            return 10
        return None
    assert len(asset_2.input_names) == 1
    assert asset_2.op.ins['asset_1'].dagster_type.is_nothing
    res = materialize([asset_1, asset_2], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_single_asset_deps_via_string():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def asset_1():
        if False:
            i = 10
            return i + 15
        return None

    @asset(deps=['asset_1'])
    def asset_2():
        if False:
            for i in range(10):
                print('nop')
        return None
    assert len(asset_2.input_names) == 1
    assert asset_2.op.ins['asset_1'].dagster_type.is_nothing
    res = materialize([asset_1, asset_2], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_single_asset_deps_via_asset_key():
    if False:
        i = 10
        return i + 15

    @asset
    def asset_1():
        if False:
            i = 10
            return i + 15
        return None

    @asset(deps=[AssetKey('asset_1')])
    def asset_2():
        if False:
            return 10
        return None
    assert len(asset_2.input_names) == 1
    assert asset_2.op.ins['asset_1'].dagster_type.is_nothing
    res = materialize([asset_1, asset_2], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_single_asset_deps_via_mixed_types():
    if False:
        while True:
            i = 10

    @asset
    def via_definition():
        if False:
            while True:
                i = 10
        return None

    @asset
    def via_string():
        if False:
            for i in range(10):
                print('nop')
        return None

    @asset
    def via_asset_key():
        if False:
            for i in range(10):
                print('nop')
        return None

    @asset(deps=[via_definition, 'via_string', AssetKey('via_asset_key')])
    def downstream():
        if False:
            while True:
                i = 10
        return None
    assert len(downstream.input_names) == 3
    assert downstream.op.ins['via_definition'].dagster_type.is_nothing
    assert downstream.op.ins['via_string'].dagster_type.is_nothing
    assert downstream.op.ins['via_asset_key'].dagster_type.is_nothing
    res = materialize([via_definition, via_string, via_asset_key, downstream], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_deps_via_string():
    if False:
        print('Hello World!')

    @multi_asset(outs={'asset_1': AssetOut(), 'asset_2': AssetOut()})
    def a_multi_asset():
        if False:
            i = 10
            return i + 15
        return (None, None)

    @asset(deps=['asset_1'])
    def depends_on_one_sub_asset():
        if False:
            while True:
                i = 10
        return None
    assert len(depends_on_one_sub_asset.input_names) == 1
    assert depends_on_one_sub_asset.op.ins['asset_1'].dagster_type.is_nothing

    @asset(deps=['asset_1', 'asset_2'])
    def depends_on_both_sub_assets():
        if False:
            while True:
                i = 10
        return None
    assert len(depends_on_both_sub_assets.input_names) == 2
    assert depends_on_both_sub_assets.op.ins['asset_1'].dagster_type.is_nothing
    assert depends_on_both_sub_assets.op.ins['asset_2'].dagster_type.is_nothing
    res = materialize([a_multi_asset, depends_on_one_sub_asset, depends_on_both_sub_assets], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_deps_via_key():
    if False:
        for i in range(10):
            print('nop')

    @multi_asset(outs={'asset_1': AssetOut(), 'asset_2': AssetOut()})
    def a_multi_asset():
        if False:
            return 10
        return (None, None)

    @asset(deps=[AssetKey('asset_1')])
    def depends_on_one_sub_asset():
        if False:
            while True:
                i = 10
        return None
    assert len(depends_on_one_sub_asset.input_names) == 1
    assert depends_on_one_sub_asset.op.ins['asset_1'].dagster_type.is_nothing

    @asset(deps=[AssetKey('asset_1'), AssetKey('asset_2')])
    def depends_on_both_sub_assets():
        if False:
            print('Hello World!')
        return None
    assert len(depends_on_both_sub_assets.input_names) == 2
    assert depends_on_both_sub_assets.op.ins['asset_1'].dagster_type.is_nothing
    assert depends_on_both_sub_assets.op.ins['asset_2'].dagster_type.is_nothing
    res = materialize([a_multi_asset, depends_on_one_sub_asset, depends_on_both_sub_assets], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_deps_via_mixed_types():
    if False:
        print('Hello World!')

    @multi_asset(outs={'asset_1': AssetOut(), 'asset_2': AssetOut()})
    def a_multi_asset():
        if False:
            print('Hello World!')
        return (None, None)

    @asset(deps=[AssetKey('asset_1'), 'asset_2'])
    def depends_on_both_sub_assets():
        if False:
            print('Hello World!')
        return None
    assert len(depends_on_both_sub_assets.input_names) == 2
    assert depends_on_both_sub_assets.op.ins['asset_1'].dagster_type.is_nothing
    assert depends_on_both_sub_assets.op.ins['asset_2'].dagster_type.is_nothing
    res = materialize([a_multi_asset, depends_on_both_sub_assets], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_deps_with_set():
    if False:
        for i in range(10):
            print('nop')

    @multi_asset(outs={'asset_1': AssetOut(), 'asset_2': AssetOut()})
    def a_multi_asset():
        if False:
            i = 10
            return i + 15
        return (None, None)

    @asset(deps=set(['asset_1', 'asset_2']))
    def depends_on_both_sub_assets():
        if False:
            for i in range(10):
                print('nop')
        return None
    assert len(depends_on_both_sub_assets.input_names) == 2
    assert depends_on_both_sub_assets.op.ins['asset_1'].dagster_type.is_nothing
    assert depends_on_both_sub_assets.op.ins['asset_2'].dagster_type.is_nothing
    res = materialize([a_multi_asset, depends_on_both_sub_assets], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_deps_via_assets_definition():
    if False:
        i = 10
        return i + 15

    @multi_asset(outs={'asset_1': AssetOut(), 'asset_2': AssetOut()})
    def a_multi_asset():
        if False:
            for i in range(10):
                print('nop')
        return (None, None)

    @asset(deps=[a_multi_asset])
    def depends_on_both_sub_assets():
        if False:
            print('Hello World!')
        return None
    assert len(depends_on_both_sub_assets.input_names) == 2
    assert depends_on_both_sub_assets.op.ins['asset_1'].dagster_type.is_nothing
    assert depends_on_both_sub_assets.op.ins['asset_2'].dagster_type.is_nothing
    res = materialize([a_multi_asset, depends_on_both_sub_assets], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_downstream_deps_via_assets_definition():
    if False:
        print('Hello World!')

    @asset
    def asset_1():
        if False:
            i = 10
            return i + 15
        return None

    @multi_asset(deps=[asset_1], outs={'out1': AssetOut(), 'out2': AssetOut()})
    def asset_2():
        if False:
            while True:
                i = 10
        return (None, None)
    assert len(asset_2.input_names) == 1
    assert asset_2.op.ins['asset_1'].dagster_type.is_nothing
    res = materialize([asset_1, asset_2], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_downstream_deps_via_string():
    if False:
        i = 10
        return i + 15

    @asset
    def asset_1():
        if False:
            print('Hello World!')
        return None

    @multi_asset(deps=['asset_1'], outs={'out1': AssetOut(), 'out2': AssetOut()})
    def asset_2():
        if False:
            while True:
                i = 10
        return (None, None)
    assert len(asset_2.input_names) == 1
    assert asset_2.op.ins['asset_1'].dagster_type.is_nothing
    res = materialize([asset_1, asset_2], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_downstream_deps_via_asset_key():
    if False:
        while True:
            i = 10

    @asset
    def asset_1():
        if False:
            return 10
        return None

    @multi_asset(deps=[AssetKey('asset_1')], outs={'out1': AssetOut(), 'out2': AssetOut()})
    def asset_2():
        if False:
            return 10
        return (None, None)
    assert len(asset_2.input_names) == 1
    assert asset_2.op.ins['asset_1'].dagster_type.is_nothing
    res = materialize([asset_1, asset_2], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_multi_asset_downstream_deps_via_mixed_types():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def via_definition():
        if False:
            return 10
        return None

    @asset
    def via_string():
        if False:
            print('Hello World!')
        return None

    @asset
    def via_asset_key():
        if False:
            while True:
                i = 10
        return None

    @multi_asset(deps=[via_definition, 'via_string', AssetKey('via_asset_key')], outs={'out1': AssetOut(), 'out2': AssetOut()})
    def downstream():
        if False:
            while True:
                i = 10
        return (None, None)
    assert len(downstream.input_names) == 3
    assert downstream.op.ins['via_definition'].dagster_type.is_nothing
    assert downstream.op.ins['via_string'].dagster_type.is_nothing
    assert downstream.op.ins['via_asset_key'].dagster_type.is_nothing
    res = materialize([via_definition, via_string, via_asset_key, downstream], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_source_asset_deps_via_assets_definition():
    if False:
        i = 10
        return i + 15
    a_source_asset = SourceAsset('a_key')

    @asset(deps=[a_source_asset])
    def depends_on_source_asset():
        if False:
            for i in range(10):
                print('nop')
        return None
    assert len(depends_on_source_asset.input_names) == 1
    assert depends_on_source_asset.op.ins['a_key'].dagster_type.is_nothing
    res = materialize([depends_on_source_asset], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_source_asset_deps_via_string():
    if False:
        print('Hello World!')
    a_source_asset = SourceAsset('a_key')

    @asset(deps=['a_key'])
    def depends_on_source_asset():
        if False:
            print('Hello World!')
        return None
    assert len(depends_on_source_asset.input_names) == 1
    assert depends_on_source_asset.op.ins['a_key'].dagster_type.is_nothing
    res = materialize([depends_on_source_asset], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_source_asset_deps_via_key():
    if False:
        return 10
    a_source_asset = SourceAsset('a_key')

    @asset(deps=[AssetKey('a_key')])
    def depends_on_source_asset():
        if False:
            return 10
        return None
    assert len(depends_on_source_asset.input_names) == 1
    assert depends_on_source_asset.op.ins['a_key'].dagster_type.is_nothing
    res = materialize([depends_on_source_asset], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_interop():
    if False:
        while True:
            i = 10

    @asset
    def no_value_asset():
        if False:
            i = 10
            return i + 15
        return None

    @asset(io_manager_key='fs_io_manager')
    def value_asset() -> int:
        if False:
            for i in range(10):
                print('nop')
        return 1

    @asset(deps=[no_value_asset])
    def interop_asset(value_asset: int):
        if False:
            while True:
                i = 10
        assert value_asset == 1
    assert len(interop_asset.input_names) == 2
    assert interop_asset.op.ins['no_value_asset'].dagster_type.is_nothing
    assert interop_asset.op.ins['value_asset'].dagster_type.kind == DagsterTypeKind.SCALAR
    res = materialize([no_value_asset, value_asset, interop_asset], resources={'io_manager': TestingIOManager(), 'fs_io_manager': FilesystemIOManager()})
    assert res.success

def test_non_existent_asset_key():
    if False:
        while True:
            i = 10

    @asset(deps=['not_real'])
    def my_asset():
        if False:
            print('Hello World!')
        return None
    res = materialize([my_asset], resources={'io_manager': TestingIOManager()})
    assert res.success

def test_bad_types():
    if False:
        return 10

    class NotAnAsset:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.foo = 'bar'
    not_an_asset = NotAnAsset()
    with pytest.raises(ParameterCheckError, match='Param "asset" is not one of '):

        @asset(deps=[not_an_asset])
        def my_asset():
            if False:
                for i in range(10):
                    print('nop')
            return None

def test_dep_via_deps_and_fn():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def the_upstream_asset():
        if False:
            for i in range(10):
                print('nop')
        return 1
    with pytest.raises(DagsterInvalidDefinitionError, match='deps value .* also declared as input/AssetIn'):

        @asset(deps=[the_upstream_asset])
        def depends_on_upstream_asset(the_upstream_asset):
            if False:
                return 10
            return None

def test_duplicate_deps():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def the_upstream_asset():
        if False:
            while True:
                i = 10
        return None

    @asset(deps=[the_upstream_asset, the_upstream_asset])
    def the_downstream_asset():
        if False:
            while True:
                i = 10
        return None
    assert len(the_downstream_asset.input_names) == 1
    assert the_downstream_asset.op.ins['the_upstream_asset'].dagster_type.is_nothing
    res = materialize([the_downstream_asset, the_upstream_asset], resources={'io_manager': TestingIOManager(), 'fs_io_manager': FilesystemIOManager()})
    assert res.success
    with pytest.raises(DagsterInvariantViolationError, match='Cannot set a dependency on asset .* more than once'):

        @asset(deps=[the_upstream_asset, AssetDep(asset=the_upstream_asset, partition_mapping=TimeWindowPartitionMapping(start_offset=-1, end_offset=-1))])
        def conflicting_deps():
            if False:
                while True:
                    i = 10
            return None