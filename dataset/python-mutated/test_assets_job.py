import hashlib
import os
import warnings
import pytest
from dagster import AssetKey, AssetOut, AssetsDefinition, DagsterEventType, DagsterInvalidDefinitionError, DailyPartitionsDefinition, Definitions, DependencyDefinition, Field, GraphIn, GraphOut, HourlyPartitionsDefinition, In, IOManager, Nothing, Out, Output, ResourceDefinition, StaticPartitionsDefinition, build_reconstructable_job, define_asset_job, graph, io_manager, materialize_to_memory, multi_asset, observable_source_asset, op, resource, with_resources
from dagster._config import StringSource
from dagster._core.definitions import AssetIn, SourceAsset, asset, build_assets_job
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.asset_selection import AssetSelection, CoercibleToAssetSelection
from dagster._core.definitions.assets_job import get_base_asset_jobs
from dagster._core.definitions.dependency import NodeHandle, NodeInvocation
from dagster._core.definitions.executor_definition import in_process_executor
from dagster._core.definitions.load_assets_from_modules import prefix_assets
from dagster._core.errors import DagsterInvalidSubsetError
from dagster._core.execution.api import execute_run_iterator
from dagster._core.snap import DependencyStructureIndex
from dagster._core.snap.dep_snapshot import OutputHandleSnap, build_dep_structure_snapshot_from_graph_def
from dagster._core.storage.event_log.base import EventRecordsFilter
from dagster._core.test_utils import ignore_warning, instance_for_test
from dagster._utils import safe_tempfile_path
from dagster._utils.warnings import disable_dagster_warnings

@pytest.fixture(autouse=True)
def error_on_warning():
    if False:
        while True:
            i = 10
    warnings.resetwarnings()
    warnings.filterwarnings('error')

def _all_asset_keys(result):
    if False:
        for i in range(10):
            print('nop')
    mats = [event.event_specific_data.materialization for event in result.all_events if event.event_type_value == 'ASSET_MATERIALIZATION']
    ret = {mat.asset_key for mat in mats}
    assert len(mats) == len(ret)
    return ret

def _asset_keys_for_node(result, node_name):
    if False:
        return 10
    mats = result.asset_materializations_for_node(node_name)
    ret = {mat.asset_key for mat in mats}
    assert len(mats) == len(ret)
    return ret

def test_single_asset_job():
    if False:
        print('Hello World!')

    @asset
    def asset1(context):
        if False:
            return 10
        assert context.asset_key == AssetKey(['asset1'])
        return 1
    job = build_assets_job('a', [asset1])
    assert job.graph.node_defs == [asset1.op]
    assert job.execute_in_process().success

def test_two_asset_job():
    if False:
        i = 10
        return i + 15

    @asset
    def asset1():
        if False:
            print('Hello World!')
        return 1

    @asset
    def asset2(asset1):
        if False:
            return 10
        assert asset1 == 1
    job = build_assets_job('a', [asset1, asset2])
    assert job.graph.node_defs == [asset1.op, asset2.op]
    assert job.dependencies == {NodeInvocation('asset1'): {}, NodeInvocation('asset2'): {'asset1': DependencyDefinition('asset1', 'result')}}
    assert job.execute_in_process().success

def test_single_asset_job_with_config():
    if False:
        return 10

    @asset(config_schema={'foo': Field(StringSource)})
    def asset1(context):
        if False:
            return 10
        return context.op_config['foo']
    job = build_assets_job('a', [asset1])
    assert job.graph.node_defs == [asset1.op]
    assert job.execute_in_process(run_config={'ops': {'asset1': {'config': {'foo': 'bar'}}}}).success

def test_fork():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def asset1():
        if False:
            return 10
        return 1

    @asset
    def asset2(asset1):
        if False:
            print('Hello World!')
        assert asset1 == 1

    @asset
    def asset3(asset1):
        if False:
            print('Hello World!')
        assert asset1 == 1
    job = build_assets_job('a', [asset1, asset2, asset3])
    assert job.graph.node_defs == [asset1.op, asset2.op, asset3.op]
    assert job.dependencies == {NodeInvocation('asset1'): {}, NodeInvocation('asset2'): {'asset1': DependencyDefinition('asset1', 'result')}, NodeInvocation('asset3'): {'asset1': DependencyDefinition('asset1', 'result')}}
    assert job.execute_in_process().success

def test_join():
    if False:
        while True:
            i = 10

    @asset
    def asset1():
        if False:
            return 10
        return 1

    @asset
    def asset2():
        if False:
            i = 10
            return i + 15
        return 2

    @asset
    def asset3(asset1, asset2):
        if False:
            return 10
        assert asset1 == 1
        assert asset2 == 2
    job = build_assets_job('a', [asset1, asset2, asset3])
    assert job.graph.node_defs == [asset1.op, asset2.op, asset3.op]
    assert job.dependencies == {NodeInvocation('asset1'): {}, NodeInvocation('asset2'): {}, NodeInvocation('asset3'): {'asset1': DependencyDefinition('asset1', 'result'), 'asset2': DependencyDefinition('asset2', 'result')}}
    result = job.execute_in_process()
    assert _asset_keys_for_node(result, 'asset3') == {AssetKey('asset3')}

def test_asset_key_output():
    if False:
        return 10

    @asset
    def asset1():
        if False:
            i = 10
            return i + 15
        return 1

    @asset(ins={'hello': AssetIn(key=AssetKey('asset1'))})
    def asset2(hello):
        if False:
            for i in range(10):
                print('nop')
        return hello
    job = build_assets_job('boo', [asset1, asset2])
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('asset2') == 1
    assert _asset_keys_for_node(result, 'asset2') == {AssetKey('asset2')}

def test_asset_key_matches_input_name():
    if False:
        return 10

    @asset
    def asset_foo():
        if False:
            i = 10
            return i + 15
        return 'foo'

    @asset
    def asset_bar():
        if False:
            for i in range(10):
                print('nop')
        return 'bar'

    @asset(ins={'asset_bar': AssetIn(key=AssetKey('asset_foo'))})
    def last_asset(asset_bar):
        if False:
            for i in range(10):
                print('nop')
        return asset_bar
    job = build_assets_job('lol', [asset_foo, asset_bar, last_asset])
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('last_asset') == 'foo'
    assert _asset_keys_for_node(result, 'last_asset') == {AssetKey('last_asset')}

def test_asset_key_and_inferred():
    if False:
        print('Hello World!')

    @asset
    def asset_foo():
        if False:
            while True:
                i = 10
        return 2

    @asset
    def asset_bar():
        if False:
            print('Hello World!')
        return 5

    @asset(ins={'foo': AssetIn(key=AssetKey('asset_foo'))})
    def asset_baz(foo, asset_bar):
        if False:
            for i in range(10):
                print('nop')
        return foo + asset_bar
    job = build_assets_job('hello', [asset_foo, asset_bar, asset_baz])
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('asset_baz') == 7
    assert _asset_keys_for_node(result, 'asset_baz') == {AssetKey('asset_baz')}

def test_asset_key_for_asset_with_key_prefix_str():
    if False:
        for i in range(10):
            print('nop')

    @asset(key_prefix='hello')
    def asset_foo():
        if False:
            return 10
        return 'foo'

    @asset(ins={'foo': AssetIn(key=AssetKey(['hello', 'asset_foo']))})
    def success_asset(foo):
        if False:
            while True:
                i = 10
        return foo
    job = build_assets_job('lol', [asset_foo, success_asset])
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('success_asset') == 'foo'
    assert _asset_keys_for_node(result, 'hello__asset_foo') == {AssetKey(['hello', 'asset_foo'])}

def test_source_asset():
    if False:
        return 10

    @asset
    def asset1(source1):
        if False:
            return 10
        assert source1 == 5
        return 1

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def load_input(self, context):
            if False:
                return 10
            assert context.resource_config['a'] == 7
            assert context.resources.subresource == 9
            assert context.upstream_output.resources.subresource == 9
            assert context.upstream_output.asset_key == AssetKey('source1')
            assert context.upstream_output.metadata == {'a': 'b'}
            assert context.upstream_output.resource_config['a'] == 7
            assert context.upstream_output.log is not None
            context.upstream_output.log.info('hullo')
            assert context.asset_key == AssetKey('source1')
            return 5

    @io_manager(config_schema={'a': int}, required_resource_keys={'subresource'})
    def my_io_manager(_):
        if False:
            return 10
        return MyIOManager()
    job = build_assets_job('a', [asset1], source_assets=[SourceAsset(AssetKey('source1'), io_manager_key='special_io_manager', metadata={'a': 'b'})], resource_defs={'special_io_manager': my_io_manager.configured({'a': 7}), 'subresource': ResourceDefinition.hardcoded_resource(9)})
    assert job.graph.node_defs == [asset1.op]
    result = job.execute_in_process()
    assert result.success
    assert _asset_keys_for_node(result, 'asset1') == {AssetKey('asset1')}

def test_missing_io_manager():
    if False:
        return 10

    @asset
    def asset1(source1):
        if False:
            for i in range(10):
                print('nop')
        return source1
    with pytest.raises(DagsterInvalidDefinitionError, match='io manager with key \'special_io_manager\' required by SourceAsset with key \\[\\"source1\\"\\] was not provided.'):
        build_assets_job('a', [asset1], source_assets=[SourceAsset(AssetKey('source1'), io_manager_key='special_io_manager')])

def test_source_op_asset():
    if False:
        i = 10
        return i + 15

    @asset(io_manager_key='special_io_manager')
    def source1():
        if False:
            i = 10
            return i + 15
        pass

    @asset
    def asset1(source1):
        if False:
            return 10
        assert source1 == 5
        return 1

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                i = 10
                return i + 15
            pass

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            return 5

    @io_manager
    def my_io_manager(_):
        if False:
            i = 10
            return i + 15
        return MyIOManager()
    job = build_assets_job('a', [asset1], source_assets=[source1], resource_defs={'special_io_manager': my_io_manager})
    assert job.graph.node_defs == [asset1.op]
    result = job.execute_in_process()
    assert result.success
    assert _asset_keys_for_node(result, 'asset1') == {AssetKey('asset1')}

def test_deps():
    if False:
        return 10
    with safe_tempfile_path() as path:

        @asset
        def foo():
            if False:
                return 10
            with open(path, 'w', encoding='utf8') as ff:
                ff.write('yup')

        @asset(deps=[AssetKey('foo')])
        def bar():
            if False:
                while True:
                    i = 10
            assert os.path.exists(path)
        job = build_assets_job('a', [foo, bar])
        result = job.execute_in_process()
        assert result.success
        assert _asset_keys_for_node(result, 'foo') == {AssetKey('foo')}
        assert _asset_keys_for_node(result, 'bar') == {AssetKey('bar')}

def test_deps_as_str():
    if False:
        print('Hello World!')

    @asset
    def foo():
        if False:
            i = 10
            return i + 15
        pass

    @asset(deps=['foo'])
    def bar():
        if False:
            i = 10
            return i + 15
        pass
    assert AssetKey('foo') in bar.asset_deps[AssetKey('bar')]

def test_multiple_deps():
    if False:
        return 10

    @asset
    def foo():
        if False:
            print('Hello World!')
        pass

    @asset(key_prefix='key_prefix')
    def bar():
        if False:
            print('Hello World!')
        pass

    @asset
    def baz():
        if False:
            print('Hello World!')
        return 1

    @asset(deps=[AssetKey('foo'), AssetKey(['key_prefix', 'bar'])])
    def qux(baz):
        if False:
            return 10
        return baz
    job = build_assets_job('a', [foo, bar, baz, qux])
    dep_structure_snapshot = build_dep_structure_snapshot_from_graph_def(job.graph)
    index = DependencyStructureIndex(dep_structure_snapshot)
    assert index.get_invocation('foo')
    assert index.get_invocation('key_prefix__bar')
    assert index.get_invocation('baz')
    assert index.get_upstream_outputs('qux', 'foo') == [OutputHandleSnap('foo', 'result')]
    assert index.get_upstream_outputs('qux', 'key_prefix_bar') == [OutputHandleSnap('key_prefix__bar', 'result')]
    assert index.get_upstream_outputs('qux', 'baz') == [OutputHandleSnap('baz', 'result')]
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('qux') == 1
    assert _asset_keys_for_node(result, 'key_prefix__bar') == {AssetKey(['key_prefix', 'bar'])}
    assert _asset_keys_for_node(result, 'qux') == {AssetKey('qux')}

def test_basic_graph_asset():
    if False:
        i = 10
        return i + 15

    @op
    def return_one():
        if False:
            print('Hello World!')
        return 1

    @op
    def add_one(in1):
        if False:
            return 10
        return in1 + 1

    @graph
    def create_cool_thing():
        if False:
            while True:
                i = 10
        return add_one(add_one(return_one()))
    cool_thing_asset = AssetsDefinition(keys_by_input_name={}, keys_by_output_name={'result': AssetKey('cool_thing')}, node_def=create_cool_thing)
    job = build_assets_job('graph_asset_job', [cool_thing_asset])
    result = job.execute_in_process()
    assert _asset_keys_for_node(result, 'create_cool_thing.add_one_2') == {AssetKey('cool_thing')}

def test_input_mapped_graph_asset():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def a():
        if False:
            for i in range(10):
                print('nop')
        return 'a'

    @asset
    def b():
        if False:
            print('Hello World!')
        return 'b'

    @op
    def double_string(s):
        if False:
            print('Hello World!')
        return s * 2

    @op
    def combine_strings(s1, s2):
        if False:
            i = 10
            return i + 15
        return s1 + s2

    @graph
    def create_cool_thing(a, b):
        if False:
            for i in range(10):
                print('nop')
        da = double_string(double_string(a))
        db = double_string(b)
        return combine_strings(da, db)
    cool_thing_asset = AssetsDefinition(keys_by_input_name={'a': AssetKey('a'), 'b': AssetKey('b')}, keys_by_output_name={'result': AssetKey('cool_thing')}, node_def=create_cool_thing)
    job = build_assets_job('graph_asset_job', [a, b, cool_thing_asset])
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('create_cool_thing.combine_strings') == 'aaaabb'
    assert _asset_keys_for_node(result, 'create_cool_thing') == {AssetKey('cool_thing')}
    assert _asset_keys_for_node(result, 'create_cool_thing.combine_strings') == {AssetKey('cool_thing')}

def test_output_mapped_same_op_graph_asset():
    if False:
        print('Hello World!')

    @asset
    def a():
        if False:
            print('Hello World!')
        return 'a'

    @asset
    def b():
        if False:
            return 10
        return 'b'

    @op
    def double_string(s):
        if False:
            print('Hello World!')
        return s * 2

    @op(out={'ns1': Out(), 'ns2': Out()})
    def combine_strings_and_split(s1, s2):
        if False:
            for i in range(10):
                print('nop')
        return (s1 + s2, s2 + s1)

    @graph(out={'o1': GraphOut(), 'o2': GraphOut()})
    def create_cool_things(a, b):
        if False:
            return 10
        da = double_string(double_string(a))
        db = double_string(b)
        (o1, o2) = combine_strings_and_split(da, db)
        return (o1, o2)

    @asset
    def out_asset1_plus_one(out_asset1):
        if False:
            return 10
        return out_asset1 + 'one'

    @asset
    def out_asset2_plus_one(out_asset2):
        if False:
            print('Hello World!')
        return out_asset2 + 'one'
    complex_asset = AssetsDefinition(keys_by_input_name={'a': AssetKey('a'), 'b': AssetKey('b')}, keys_by_output_name={'o1': AssetKey('out_asset1'), 'o2': AssetKey('out_asset2')}, node_def=create_cool_things)
    job = build_assets_job('graph_asset_job', [a, b, complex_asset, out_asset1_plus_one, out_asset2_plus_one])
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('out_asset1_plus_one') == 'aaaabbone'
    assert result.output_for_node('out_asset2_plus_one') == 'bbaaaaone'
    assert _asset_keys_for_node(result, 'create_cool_things') == {AssetKey('out_asset1'), AssetKey('out_asset2')}
    assert _asset_keys_for_node(result, 'create_cool_things.combine_strings_and_split') == {AssetKey('out_asset1'), AssetKey('out_asset2')}

def test_output_mapped_different_op_graph_asset():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def a():
        if False:
            while True:
                i = 10
        return 'a'

    @asset
    def b():
        if False:
            i = 10
            return i + 15
        return 'b'

    @op
    def double_string(s):
        if False:
            for i in range(10):
                print('nop')
        return s * 2

    @op(out={'ns1': Out(), 'ns2': Out()})
    def combine_strings_and_split(s1, s2):
        if False:
            while True:
                i = 10
        return (s1 + s2, s2 + s1)

    @graph(out={'o1': GraphOut(), 'o2': GraphOut()})
    def create_cool_things(a, b):
        if False:
            i = 10
            return i + 15
        (ab, ba) = combine_strings_and_split(a, b)
        dab = double_string(ab)
        dba = double_string(ba)
        return (dab, dba)

    @asset
    def out_asset1_plus_one(out_asset1):
        if False:
            print('Hello World!')
        return out_asset1 + 'one'

    @asset
    def out_asset2_plus_one(out_asset2):
        if False:
            for i in range(10):
                print('nop')
        return out_asset2 + 'one'
    complex_asset = AssetsDefinition(keys_by_input_name={'a': AssetKey('a'), 'b': AssetKey('b')}, keys_by_output_name={'o1': AssetKey('out_asset1'), 'o2': AssetKey('out_asset2')}, node_def=create_cool_things)
    job = build_assets_job('graph_asset_job', [a, b, complex_asset, out_asset1_plus_one, out_asset2_plus_one])
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('out_asset1_plus_one') == 'ababone'
    assert result.output_for_node('out_asset2_plus_one') == 'babaone'
    assert _asset_keys_for_node(result, 'create_cool_things') == {AssetKey('out_asset1'), AssetKey('out_asset2')}
    assert _asset_keys_for_node(result, 'create_cool_things.double_string') == {AssetKey('out_asset1')}
    assert _asset_keys_for_node(result, 'create_cool_things.double_string_2') == {AssetKey('out_asset2')}

def test_nasty_nested_graph_assets():
    if False:
        i = 10
        return i + 15

    @op
    def add_one(i):
        if False:
            for i in range(10):
                print('nop')
        return i + 1

    @graph
    def add_three(i):
        if False:
            return 10
        return add_one(add_one(add_one(i)))

    @graph
    def add_five(i):
        if False:
            while True:
                i = 10
        return add_one(add_three(add_one(i)))

    @op
    def get_sum(a, b):
        if False:
            while True:
                i = 10
        return a + b

    @graph
    def sum_plus_one(a, b):
        if False:
            for i in range(10):
                print('nop')
        return add_one(get_sum(a, b))

    @asset
    def zero():
        if False:
            return 10
        return 0

    @graph(out={'eight': GraphOut(), 'five': GraphOut()})
    def create_eight_and_five(zero):
        if False:
            print('Hello World!')
        return (add_five(add_three(zero)), add_five(zero))

    @graph(out={'thirteen': GraphOut(), 'six': GraphOut()})
    def create_thirteen_and_six(eight, five, zero):
        if False:
            print('Hello World!')
        return (add_five(eight), sum_plus_one(five, zero))

    @graph
    def create_twenty(thirteen, six):
        if False:
            print('Hello World!')
        return sum_plus_one(thirteen, six)
    eight_and_five = AssetsDefinition(keys_by_input_name={'zero': AssetKey('zero')}, keys_by_output_name={'eight': AssetKey('eight'), 'five': AssetKey('five')}, node_def=create_eight_and_five)
    thirteen_and_six = AssetsDefinition(keys_by_input_name={'eight': AssetKey('eight'), 'five': AssetKey('five'), 'zero': AssetKey('zero')}, keys_by_output_name={'thirteen': AssetKey('thirteen'), 'six': AssetKey('six')}, node_def=create_thirteen_and_six)
    twenty = AssetsDefinition(keys_by_input_name={'thirteen': AssetKey('thirteen'), 'six': AssetKey('six')}, keys_by_output_name={'result': AssetKey('twenty')}, node_def=create_twenty)
    job = build_assets_job('graph_asset_job', [zero, eight_and_five, thirteen_and_six, twenty])
    result = job.execute_in_process()
    assert result.success
    assert result.output_for_node('create_thirteen_and_six', 'six') == 6
    assert result.output_for_node('create_twenty') == 20
    assert _asset_keys_for_node(result, 'create_eight_and_five') == {AssetKey('eight'), AssetKey('five')}
    assert _asset_keys_for_node(result, 'create_thirteen_and_six') == {AssetKey('thirteen'), AssetKey('six')}
    assert _asset_keys_for_node(result, 'create_twenty') == {AssetKey('twenty')}

def test_internal_asset_deps():
    if False:
        while True:
            i = 10

    @op
    def my_op(x, y):
        if False:
            for i in range(10):
                print('nop')
        return x
    with pytest.raises(Exception, match='output_name non_exist_output_name'):

        @graph(ins={'x': GraphIn()})
        def my_graph(x, y):
            if False:
                while True:
                    i = 10
            my_op(x, y)
        AssetsDefinition.from_graph(graph_def=my_graph, internal_asset_deps={'non_exist_output_name': {AssetKey('b')}})
    with pytest.raises(Exception, match='output_name non_exist_output_name'):
        AssetsDefinition.from_op(op_def=my_op, internal_asset_deps={'non_exist_output_name': {AssetKey('b')}})

def test_asset_def_from_op_inputs():
    if False:
        while True:
            i = 10

    @op(ins={'my_input': In(), 'other_input': In()}, out={'out1': Out(), 'out2': Out()})
    def my_op(my_input, other_input):
        if False:
            for i in range(10):
                print('nop')
        pass
    assets_def = AssetsDefinition.from_op(op_def=my_op, keys_by_input_name={'my_input': AssetKey('x_asset'), 'other_input': AssetKey('y_asset')})
    assert assets_def.keys_by_input_name['my_input'] == AssetKey('x_asset')
    assert assets_def.keys_by_input_name['other_input'] == AssetKey('y_asset')
    assert assets_def.keys_by_output_name['out1'] == AssetKey('out1')
    assert assets_def.keys_by_output_name['out2'] == AssetKey('out2')

def test_asset_def_from_op_outputs():
    if False:
        return 10

    @op(ins={'my_input': In(), 'other_input': In()}, out={'out1': Out(), 'out2': Out()})
    def x_op(my_input, other_input):
        if False:
            return 10
        pass
    assets_def = AssetsDefinition.from_op(op_def=x_op, keys_by_output_name={'out2': AssetKey('y_asset'), 'out1': AssetKey('x_asset')})
    assert assets_def.keys_by_output_name['out2'] == AssetKey('y_asset')
    assert assets_def.keys_by_output_name['out1'] == AssetKey('x_asset')
    assert assets_def.keys_by_input_name['my_input'] == AssetKey('my_input')
    assert assets_def.keys_by_input_name['other_input'] == AssetKey('other_input')

def test_asset_from_op_no_args():
    if False:
        while True:
            i = 10

    @op
    def my_op(x, y):
        if False:
            print('Hello World!')
        return x
    assets_def = AssetsDefinition.from_op(op_def=my_op)
    assert assets_def.keys_by_input_name['x'] == AssetKey('x')
    assert assets_def.keys_by_input_name['y'] == AssetKey('y')
    assert assets_def.keys_by_output_name['result'] == AssetKey('my_op')

def test_asset_def_from_graph_inputs():
    if False:
        print('Hello World!')

    @op
    def my_op(x, y):
        if False:
            i = 10
            return i + 15
        return x

    @graph(ins={'x': GraphIn(), 'y': GraphIn()})
    def my_graph(x, y):
        if False:
            while True:
                i = 10
        return my_op(x, y)
    assets_def = AssetsDefinition.from_graph(graph_def=my_graph, keys_by_input_name={'x': AssetKey('x_asset'), 'y': AssetKey('y_asset')})
    assert assets_def.keys_by_input_name['x'] == AssetKey('x_asset')
    assert assets_def.keys_by_input_name['y'] == AssetKey('y_asset')

def test_asset_def_from_graph_outputs():
    if False:
        return 10

    @op
    def x_op(x):
        if False:
            for i in range(10):
                print('nop')
        return x

    @op
    def y_op(y):
        if False:
            while True:
                i = 10
        return y

    @graph(out={'x': GraphOut(), 'y': GraphOut()})
    def my_graph(x, y):
        if False:
            i = 10
            return i + 15
        return {'x': x_op(x), 'y': y_op(y)}
    assets_def = AssetsDefinition.from_graph(graph_def=my_graph, keys_by_output_name={'y': AssetKey('y_asset'), 'x': AssetKey('x_asset')})
    assert assets_def.keys_by_output_name['y'] == AssetKey('y_asset')
    assert assets_def.keys_by_output_name['x'] == AssetKey('x_asset')

def test_graph_asset_decorator_no_args():
    if False:
        i = 10
        return i + 15

    @op
    def my_op(x, y):
        if False:
            print('Hello World!')
        return x

    @graph
    def my_graph(x, y):
        if False:
            print('Hello World!')
        return my_op(x, y)
    assets_def = AssetsDefinition.from_graph(graph_def=my_graph)
    assert assets_def.keys_by_input_name['x'] == AssetKey('x')
    assert assets_def.keys_by_input_name['y'] == AssetKey('y')
    assert assets_def.keys_by_output_name['result'] == AssetKey('my_graph')

def test_graph_asset_group_name():
    if False:
        while True:
            i = 10

    @op
    def my_op1(x):
        if False:
            print('Hello World!')
        return x

    @op
    def my_op2(y):
        if False:
            for i in range(10):
                print('nop')
        return y

    @graph
    def my_graph(x):
        if False:
            return 10
        return my_op2(my_op1(x))
    assets_def = AssetsDefinition.from_graph(graph_def=my_graph, group_name='group1')
    assert assets_def.group_names_by_key[AssetKey('my_graph')] == 'group1'

def test_graph_asset_group_name_for_multiple_assets():
    if False:
        for i in range(10):
            print('nop')

    @op(out={'first_output': Out(), 'second_output': Out()})
    def two_outputs():
        if False:
            i = 10
            return i + 15
        return (1, 2)

    @graph(out={'first_asset': GraphOut(), 'second_asset': GraphOut()})
    def two_assets_graph():
        if False:
            i = 10
            return i + 15
        (one, two) = two_outputs()
        return {'first_asset': one, 'second_asset': two}
    two_assets = AssetsDefinition.from_graph(two_assets_graph, group_name='group2')
    two_assets_with_keys = AssetsDefinition.from_graph(two_assets_graph, keys_by_output_name={'first_asset': AssetKey('first_asset_key'), 'second_asset': AssetKey('second_asset_key')}, group_name='group3')
    assert two_assets.group_names_by_key[AssetKey('first_asset')] == 'group2'
    assert two_assets.group_names_by_key[AssetKey('second_asset')] == 'group2'
    assert two_assets_with_keys.group_names_by_key[AssetKey('first_asset_key')] == 'group3'
    assert two_assets_with_keys.group_names_by_key[AssetKey('second_asset_key')] == 'group3'

def test_execute_graph_asset():
    if False:
        return 10

    @op(out={'x': Out(), 'y': Out()})
    def x_op(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.asset_key_for_output('x') == AssetKey('x_asset')
        return (1, 2)

    @graph(out={'x': GraphOut(), 'y': GraphOut()})
    def my_graph():
        if False:
            print('Hello World!')
        (x, y) = x_op()
        return {'x': x, 'y': y}
    assets_def = AssetsDefinition.from_graph(graph_def=my_graph, keys_by_output_name={'y': AssetKey('y_asset'), 'x': AssetKey('x_asset')})
    assert materialize_to_memory([assets_def]).success

def test_graph_asset_partitioned():
    if False:
        i = 10
        return i + 15

    @op
    def my_op(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.partition_key == 'a'

    @graph
    def my_graph():
        if False:
            i = 10
            return i + 15
        return my_op()
    assets_def = AssetsDefinition.from_graph(graph_def=my_graph, partitions_def=StaticPartitionsDefinition(['a', 'b', 'c']))
    assert materialize_to_memory([assets_def], partition_key='a').success

def test_all_assets_job():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def a1():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @asset
    def a2(a1):
        if False:
            return 10
        return 2
    job = build_assets_job('graph_asset_job', [a1, a2])
    node_handle_deps_by_asset = job.asset_layer.dependency_node_handles_by_asset_key
    assert node_handle_deps_by_asset[AssetKey('a1')] == {NodeHandle('a1', parent=None)}
    assert node_handle_deps_by_asset[AssetKey('a2')] == {NodeHandle('a2', parent=None)}

def test_basic_graph():
    if False:
        while True:
            i = 10

    @op
    def get_string():
        if False:
            for i in range(10):
                print('nop')
        return 'foo'

    @op(out={'ns1': Out(), 'ns2': Out()})
    def combine_strings_and_split(s1, s2):
        if False:
            return 10
        return (s1 + s2, s2 + s1)

    @graph(out={'o1': GraphOut()})
    def thing():
        if False:
            while True:
                i = 10
        da = get_string()
        db = get_string()
        (o1, o2) = combine_strings_and_split(da, db)
        return o1

    @asset
    def out_asset1_plus_one(out_asset1):
        if False:
            for i in range(10):
                print('nop')
        return out_asset1 + 'one'

    @asset
    def out_asset2_plus_one(out_asset2):
        if False:
            i = 10
            return i + 15
        return out_asset2 + 'one'
    complex_asset = AssetsDefinition(keys_by_input_name={}, keys_by_output_name={'o1': AssetKey('out_asset1')}, node_def=thing)
    job = build_assets_job('graph_asset_job', [complex_asset])
    node_handle_deps_by_asset = job.asset_layer.dependency_node_handles_by_asset_key
    thing_handle = NodeHandle(name='thing', parent=None)
    assert node_handle_deps_by_asset[AssetKey('out_asset1')] == {NodeHandle('get_string', parent=thing_handle), NodeHandle('get_string_2', parent=thing_handle), NodeHandle('combine_strings_and_split', parent=thing_handle)}

def test_hanging_op_graph():
    if False:
        return 10

    @op
    def get_string():
        if False:
            i = 10
            return i + 15
        return 'foo'

    @op
    def combine_strings(s1, s2):
        if False:
            print('Hello World!')
        return s1 + s2

    @op
    def hanging_op():
        if False:
            i = 10
            return i + 15
        return 'bar'

    @graph(out={'o1': GraphOut(), 'o2': GraphOut()})
    def thing():
        if False:
            while True:
                i = 10
        da = get_string()
        db = get_string()
        o1 = combine_strings(da, db)
        o2 = hanging_op()
        return {'o1': o1, 'o2': o2}
    complex_asset = AssetsDefinition(keys_by_input_name={}, keys_by_output_name={'o1': AssetKey('out_asset1'), 'o2': AssetKey('out_asset2')}, node_def=thing)
    job = build_assets_job('graph_asset_job', [complex_asset])
    node_handle_deps_by_asset = job.asset_layer.dependency_node_handles_by_asset_key
    thing_handle = NodeHandle(name='thing', parent=None)
    assert node_handle_deps_by_asset[AssetKey('out_asset1')] == {NodeHandle('get_string', parent=thing_handle), NodeHandle('get_string_2', parent=thing_handle), NodeHandle('combine_strings', parent=thing_handle)}
    assert node_handle_deps_by_asset[AssetKey('out_asset2')] == {NodeHandle('hanging_op', parent=thing_handle)}

def test_nested_graph():
    if False:
        while True:
            i = 10

    @op
    def get_inside_string():
        if False:
            while True:
                i = 10
        return 'bar'

    @graph(out={'o2': GraphOut()})
    def inside_thing():
        if False:
            while True:
                i = 10
        return get_inside_string()

    @op
    def get_string():
        if False:
            while True:
                i = 10
        return 'foo'

    @op(out={'ns1': Out(), 'ns2': Out()})
    def combine_strings_and_split(s1, s2):
        if False:
            return 10
        return (s1 + s2, s2 + s1)

    @graph(out={'o1': GraphOut()})
    def thing():
        if False:
            i = 10
            return i + 15
        da = inside_thing()
        db = get_string()
        (o1, o2) = combine_strings_and_split(da, db)
        return o1
    thing_asset = AssetsDefinition(keys_by_input_name={}, keys_by_output_name={'o1': AssetKey('thing')}, node_def=thing)
    job = build_assets_job('graph_asset_job', [thing_asset])
    node_handle_deps_by_asset = job.asset_layer.dependency_node_handles_by_asset_key
    thing_handle = NodeHandle(name='thing', parent=None)
    assert node_handle_deps_by_asset[AssetKey('thing')] == {NodeHandle('get_inside_string', parent=NodeHandle('inside_thing', parent=thing_handle)), NodeHandle('get_string', parent=thing_handle), NodeHandle('combine_strings_and_split', parent=thing_handle)}

def test_asset_in_nested_graph():
    if False:
        return 10

    @op
    def get_inside_string():
        if False:
            return 10
        return 'bar'

    @op
    def get_string():
        if False:
            return 10
        return 'foo'

    @graph(out={'n1': GraphOut(), 'n2': GraphOut()})
    def inside_thing():
        if False:
            i = 10
            return i + 15
        n1 = get_inside_string()
        n2 = get_string()
        return (n1, n2)

    @op
    def get_transformed_string(string):
        if False:
            i = 10
            return i + 15
        return string + 'qux'

    @graph(out={'o1': GraphOut(), 'o3': GraphOut()})
    def thing():
        if False:
            while True:
                i = 10
        (o1, o2) = inside_thing()
        o3 = get_transformed_string(o2)
        return (o1, o3)
    thing_asset = AssetsDefinition(keys_by_input_name={}, keys_by_output_name={'o1': AssetKey('thing'), 'o3': AssetKey('thing_2')}, node_def=thing)
    job = build_assets_job('graph_asset_job', [thing_asset])
    node_handle_deps_by_asset = job.asset_layer.dependency_node_handles_by_asset_key
    thing_handle = NodeHandle(name='thing', parent=None)
    assert node_handle_deps_by_asset[AssetKey('thing')] == {NodeHandle('get_inside_string', parent=NodeHandle('inside_thing', parent=thing_handle))}
    assert node_handle_deps_by_asset[AssetKey('thing_2')] == {NodeHandle('get_string', parent=NodeHandle('inside_thing', parent=thing_handle)), NodeHandle('get_transformed_string', parent=thing_handle)}

def test_twice_nested_graph():
    if False:
        for i in range(10):
            print('nop')

    @op
    def get_inside_string():
        if False:
            return 10
        return 'bar'

    @op
    def get_string():
        if False:
            i = 10
            return i + 15
        return 'foo'

    @graph(out={'n1': GraphOut(), 'n2': GraphOut()})
    def innermost_thing():
        if False:
            i = 10
            return i + 15
        n1 = get_inside_string()
        n2 = get_string()
        return {'n1': n1, 'n2': n2}

    @op
    def transformer(string):
        if False:
            return 10
        return string + 'qux'

    @op
    def combiner(s1, s2):
        if False:
            return 10
        return s1 + s2

    @graph(out={'n1': GraphOut(), 'n2': GraphOut(), 'unused': GraphOut()})
    def middle_thing():
        if False:
            return 10
        (n1, unused_output) = innermost_thing()
        n2 = get_string()
        return {'n1': n1, 'n2': n2, 'unused': unused_output}

    @graph(out={'n1': GraphOut(), 'n2': GraphOut(), 'unused': GraphOut()})
    def outer_thing(foo_asset):
        if False:
            while True:
                i = 10
        (n1, output, unused_output) = middle_thing()
        n2 = transformer(output)
        unused_output = combiner(unused_output, transformer(foo_asset))
        return {'n1': n1, 'n2': n2, 'unused': unused_output}

    @asset
    def foo_asset():
        if False:
            print('Hello World!')
        return 'foo'
    thing_asset = AssetsDefinition.from_graph(graph_def=outer_thing, keys_by_input_name={}, keys_by_output_name={'n1': AssetKey('thing'), 'n2': AssetKey('thing_2'), 'unused': AssetKey('asjdlaksjhbdluuawubn')})
    job = build_assets_job('graph_asset_job', [foo_asset, thing_asset])
    node_handle_deps_by_asset = job.asset_layer.dependency_node_handles_by_asset_key
    outer_thing_handle = NodeHandle('outer_thing', parent=None)
    middle_thing_handle = NodeHandle('middle_thing', parent=outer_thing_handle)
    assert node_handle_deps_by_asset[AssetKey('thing')] == {NodeHandle('get_inside_string', parent=NodeHandle('innermost_thing', parent=middle_thing_handle))}
    assert node_handle_deps_by_asset[AssetKey('thing_2')] == {NodeHandle('get_string', parent=middle_thing_handle), NodeHandle('transformer', parent=outer_thing_handle)}
    assert node_handle_deps_by_asset[AssetKey('foo_asset')] == {NodeHandle('foo_asset', parent=None)}

def test_internal_asset_deps_assets():
    if False:
        while True:
            i = 10

    @op
    def upstream_op():
        if False:
            while True:
                i = 10
        return 'foo'

    @op(out={'o1': Out(), 'o2': Out()})
    def two_outputs(upstream_op):
        if False:
            for i in range(10):
                print('nop')
        o1 = upstream_op
        o2 = o1 + 'bar'
        return (o1, o2)

    @graph(out={'o1': GraphOut(), 'o2': GraphOut()})
    def thing():
        if False:
            i = 10
            return i + 15
        (o1, o2) = two_outputs(upstream_op())
        return (o1, o2)
    thing_asset = AssetsDefinition(keys_by_input_name={}, keys_by_output_name={'o1': AssetKey('thing'), 'o2': AssetKey('thing_2')}, node_def=thing, asset_deps={AssetKey('thing'): set(), AssetKey('thing_2'): {AssetKey('thing')}})

    @multi_asset(outs={'my_out_name': AssetOut(metadata={'foo': 'bar'}), 'my_other_out_name': AssetOut(metadata={'bar': 'foo'})}, internal_asset_deps={'my_out_name': {AssetKey('my_other_out_name')}, 'my_other_out_name': {AssetKey('thing')}})
    def multi_asset_with_internal_deps(thing):
        if False:
            while True:
                i = 10
        yield Output(1, 'my_out_name')
        yield Output(2, 'my_other_out_name')
    job = build_assets_job('graph_asset_job', [thing_asset, multi_asset_with_internal_deps])
    node_handle_deps_by_asset = job.asset_layer.dependency_node_handles_by_asset_key
    assert node_handle_deps_by_asset[AssetKey('thing')] == {NodeHandle('two_outputs', parent=NodeHandle('thing', parent=None)), NodeHandle(name='upstream_op', parent=NodeHandle(name='thing', parent=None))}
    assert node_handle_deps_by_asset[AssetKey('thing_2')] == {NodeHandle('two_outputs', parent=NodeHandle('thing', parent=None))}
    assert node_handle_deps_by_asset[AssetKey('my_out_name')] == {NodeHandle(name='multi_asset_with_internal_deps', parent=None)}
    assert node_handle_deps_by_asset[AssetKey('my_other_out_name')] == {NodeHandle(name='multi_asset_with_internal_deps', parent=None)}

@multi_asset(outs={'a': AssetOut(is_required=False), 'b': AssetOut(is_required=False)}, can_subset=True)
def ab(context, foo):
    if False:
        print('Hello World!')
    assert (context.selected_output_names != {'a', 'b'}) == context.is_subset
    if 'a' in context.selected_output_names:
        yield Output(foo + 1, 'a')
    if 'b' in context.selected_output_names:
        yield Output(foo + 2, 'b')

@asset
def foo():
    if False:
        for i in range(10):
            print('nop')
    return 5

@asset
def bar():
    if False:
        i = 10
        return i + 15
    return 10

@asset
def foo_bar(foo, bar):
    if False:
        while True:
            i = 10
    return foo + bar

@asset
def baz(foo_bar):
    if False:
        print('Hello World!')
    return foo_bar

@asset
def unconnected():
    if False:
        return 10
    pass
asset_defs = [foo, ab, bar, foo_bar, baz, unconnected]

def test_disconnected_subset():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        defs = Definitions(assets=asset_defs, jobs=[define_asset_job('foo')])
        foo_job = defs.get_job_def('foo')
        result = foo_job.execute_in_process(instance=instance, asset_selection=[AssetKey('unconnected'), AssetKey('bar')])
        materialization_events = [event for event in result.all_events if event.is_step_materialization]
        assert len(materialization_events) == 2
        assert materialization_events[0].asset_key == AssetKey('bar')
        assert materialization_events[1].asset_key == AssetKey('unconnected')

def test_connected_subset():
    if False:
        i = 10
        return i + 15
    with instance_for_test() as instance:
        defs = Definitions(assets=asset_defs, jobs=[define_asset_job('foo')])
        foo_job = defs.get_job_def('foo')
        result = foo_job.execute_in_process(instance=instance, asset_selection=[AssetKey('foo'), AssetKey('bar'), AssetKey('foo_bar')])
        materialization_events = sorted([event for event in result.all_events if event.is_step_materialization], key=lambda event: event.asset_key)
        assert len(materialization_events) == 3
        assert materialization_events[0].asset_key == AssetKey('bar')
        assert materialization_events[1].asset_key == AssetKey('foo')
        assert materialization_events[2].asset_key == AssetKey('foo_bar')

def test_subset_of_asset_job():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        defs = Definitions(assets=asset_defs, jobs=[define_asset_job('foo', '*baz')])
        foo_job = defs.get_job_def('foo')
        result = foo_job.execute_in_process(instance=instance, asset_selection=[AssetKey('foo'), AssetKey('bar'), AssetKey('foo_bar')])
        materialization_events = sorted([event for event in result.all_events if event.is_step_materialization], key=lambda event: event.asset_key)
        assert len(materialization_events) == 3
        assert materialization_events[0].asset_key == AssetKey('bar')
        assert materialization_events[1].asset_key == AssetKey('foo')
        assert materialization_events[2].asset_key == AssetKey('foo_bar')

def test_subset_of_build_assets_job():
    if False:
        i = 10
        return i + 15
    foo_job = build_assets_job('foo_job', assets=[foo, bar, foo_bar, baz])
    with instance_for_test() as instance:
        result = foo_job.execute_in_process(instance=instance, asset_selection=[AssetKey('foo'), AssetKey('bar'), AssetKey('foo_bar')])
        materialization_events = sorted([event for event in result.all_events if event.is_step_materialization], key=lambda event: event.asset_key)
        assert len(materialization_events) == 3
        assert materialization_events[0].asset_key == AssetKey('bar')
        assert materialization_events[1].asset_key == AssetKey('foo')
        assert materialization_events[2].asset_key == AssetKey('foo_bar')
        with pytest.raises(DagsterInvalidSubsetError):
            result = foo_job.execute_in_process(instance=instance, asset_selection=[AssetKey('unconnected')])

@resource
def my_resource():
    if False:
        i = 10
        return i + 15
    return 1

@resource
def my_resource_2():
    if False:
        print('Hello World!')
    return 1

@multi_asset(name='fivetran_sync', outs={key: AssetOut(key=AssetKey(key)) for key in ['a', 'b', 'c']})
def fivetran_asset():
    if False:
        for i in range(10):
            print('nop')
    return (1, 2, 3)

@op(name='dbt', ins={'a': In(Nothing), 'b': In(Nothing), 'c': In(Nothing)}, out={'d': Out(is_required=False), 'e': Out(is_required=False), 'f': Out(is_required=False)}, required_resource_keys={'my_resource_2'})
def dbt_op():
    if False:
        print('Hello World!')
    yield Output(4, 'f')
dbt_asset_def = AssetsDefinition(keys_by_output_name={k: AssetKey(k) for k in ['d', 'e', 'f']}, keys_by_input_name={k: AssetKey(k) for k in ['a', 'b', 'c']}, node_def=dbt_op, can_subset=True, asset_deps={AssetKey('d'): {AssetKey('a')}, AssetKey('e'): {AssetKey('d'), AssetKey('b')}, AssetKey('f'): {AssetKey('d'), AssetKey('e')}})
my_job = define_asset_job('foo', selection=['a', 'b', 'c', 'd', 'e', 'f']).resolve(asset_graph=AssetGraph.from_assets(with_resources([dbt_asset_def, fivetran_asset], resource_defs={'my_resource': my_resource, 'my_resource_2': my_resource_2})))

def reconstruct_asset_job():
    if False:
        i = 10
        return i + 15
    return my_job

def test_asset_selection_reconstructable():
    if False:
        i = 10
        return i + 15
    with disable_dagster_warnings():
        with instance_for_test() as instance:
            run = instance.create_run_for_job(job_def=my_job, asset_selection=frozenset([AssetKey('f')]))
            reconstructable_foo_job = build_reconstructable_job('dagster_tests.asset_defs_tests.test_assets_job', 'reconstruct_asset_job', reconstructable_args=tuple(), reconstructable_kwargs={}).get_subset(asset_selection=frozenset([AssetKey('f')]))
            events = list(execute_run_iterator(reconstructable_foo_job, run, instance=instance))
            assert len([event for event in events if event.is_job_success]) == 1
            materialization_planned = list(instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION_PLANNED)))
            assert len(materialization_planned) == 1

def test_job_preserved_with_asset_subset():
    if False:
        i = 10
        return i + 15

    @op(config_schema={'foo': int})
    def one(context):
        if False:
            while True:
                i = 10
        assert context.op_config['foo'] == 1
    asset_one = AssetsDefinition.from_op(one)

    @asset(config_schema={'bar': int})
    def two(context, one):
        if False:
            for i in range(10):
                print('nop')
        assert context.op_config['bar'] == 2

    @asset(config_schema={'baz': int})
    def three(context, two):
        if False:
            return 10
        assert context.op_config['baz'] == 3
    foo_job = define_asset_job('foo_job', config={'ops': {'one': {'config': {'foo': 1}}, 'two': {'config': {'bar': 2}}, 'three': {'config': {'baz': 3}}}}, description='my cool job', tags={'yay': 1}).resolve(asset_graph=AssetGraph.from_assets([asset_one, two, three]))
    result = foo_job.execute_in_process(asset_selection=[AssetKey('one')])
    assert result.success
    assert result.dagster_run.tags == {'yay': '1'}

def test_job_default_config_preserved_with_asset_subset():
    if False:
        print('Hello World!')

    @op(config_schema={'foo': Field(int, default_value=1)})
    def one(context):
        if False:
            return 10
        assert context.op_config['foo'] == 1
    asset_one = AssetsDefinition.from_op(one)

    @asset(config_schema={'bar': Field(int, default_value=2)})
    def two(context, one):
        if False:
            return 10
        assert context.op_config['bar'] == 2

    @asset(config_schema={'baz': Field(int, default_value=3)})
    def three(context, two):
        if False:
            print('Hello World!')
        assert context.op_config['baz'] == 3
    foo_job = define_asset_job('foo_job').resolve(asset_graph=AssetGraph.from_assets([asset_one, two, three]))
    result = foo_job.execute_in_process(asset_selection=[AssetKey('one')])
    assert result.success

def test_empty_asset_job():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def a():
        if False:
            while True:
                i = 10
        pass

    @asset
    def b(a):
        if False:
            return 10
        pass
    empty_selection = AssetSelection.keys('a', 'b') - AssetSelection.keys('a', 'b')
    assert empty_selection.resolve([a, b]) == set()
    empty_job = define_asset_job('empty_job', selection=empty_selection).resolve(asset_graph=AssetGraph.from_assets([a, b]))
    assert empty_job.all_node_defs == []
    result = empty_job.execute_in_process()
    assert result.success

def test_raise_error_on_incomplete_graph_asset_subset():
    if False:
        i = 10
        return i + 15

    @op
    def do_something(x):
        if False:
            return 10
        return x * 2

    @op
    def foo():
        if False:
            for i in range(10):
                print('nop')
        return (1, 2)

    @graph(out={'comments_table': GraphOut(), 'stories_table': GraphOut()})
    def complicated_graph():
        if False:
            print('Hello World!')
        result = foo()
        return (do_something(result), do_something(result))
    defs = Definitions(assets=[AssetsDefinition.from_graph(complicated_graph)], jobs=[define_asset_job('foo_job')])
    foo_job = defs.get_job_def('foo_job')
    with instance_for_test() as instance:
        with pytest.raises(DagsterInvalidSubsetError, match='complicated_graph'):
            foo_job.execute_in_process(instance=instance, asset_selection=[AssetKey('comments_table')])

def test_multi_subset():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        defs = Definitions(assets=asset_defs, jobs=[define_asset_job('foo')])
        foo_job = defs.get_job_def('foo')
        result = foo_job.execute_in_process(instance=instance, asset_selection=[AssetKey('foo'), AssetKey('a')])
        materialization_events = sorted([event for event in result.all_events if event.is_step_materialization], key=lambda event: event.asset_key)
        assert len(materialization_events) == 2
        assert materialization_events[0].asset_key == AssetKey('a')
        assert materialization_events[1].asset_key == AssetKey('foo')

def test_multi_all():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test() as instance:
        defs = Definitions(assets=asset_defs, jobs=[define_asset_job('foo')])
        foo_job = defs.get_job_def('foo')
        result = foo_job.execute_in_process(instance=instance, asset_selection=[AssetKey('foo'), AssetKey('a'), AssetKey('b')])
        materialization_events = sorted([event for event in result.all_events if event.is_step_materialization], key=lambda event: event.asset_key)
        assert len(materialization_events) == 3
        assert materialization_events[0].asset_key == AssetKey('a')
        assert materialization_events[1].asset_key == AssetKey('b')
        assert materialization_events[2].asset_key == AssetKey('foo')

def test_subset_with_source_asset():
    if False:
        i = 10
        return i + 15

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            pass

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            return 5

    @io_manager
    def the_manager():
        if False:
            i = 10
            return i + 15
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_key='the_manager')

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            i = 10
            return i + 15
        return my_source_asset + 4
    source_asset_job = Definitions(assets=[my_derived_asset, my_source_asset], resources={'the_manager': the_manager}, jobs=[define_asset_job('source_asset_job', [my_derived_asset])]).get_job_def('source_asset_job')
    result = source_asset_job.execute_in_process(asset_selection=[AssetKey('my_derived_asset')])
    assert result.success

def test_op_outputs_with_default_asset_io_mgr():
    if False:
        return 10

    @op
    def return_stuff():
        if False:
            print('Hello World!')
        return 12

    @op
    def transform(data):
        if False:
            while True:
                i = 10
        assert data == 12
        return data * 2

    @op
    def one_more_transformation(transformed_data):
        if False:
            i = 10
            return i + 15
        assert transformed_data == 24
        return transformed_data + 1

    @graph(out={'asset_1': GraphOut(), 'asset_2': GraphOut()})
    def complicated_graph():
        if False:
            while True:
                i = 10
        result = return_stuff()
        return (one_more_transformation(transform(result)), transform(result))

    @asset
    def my_asset(asset_1):
        if False:
            i = 10
            return i + 15
        assert asset_1 == 25
        return asset_1
    defs = Definitions(assets=[AssetsDefinition.from_graph(complicated_graph), my_asset], jobs=[define_asset_job('foo_job', executor_def=in_process_executor)])
    foo_job = defs.get_job_def('foo_job')
    result = foo_job.execute_in_process()
    assert result.success

def test_graph_output_is_input_within_graph():
    if False:
        i = 10
        return i + 15

    @op
    def return_stuff():
        if False:
            i = 10
            return i + 15
        return 1

    @op
    def transform(data):
        if False:
            while True:
                i = 10
        return data * 2

    @op
    def one_more_transformation(transformed_data):
        if False:
            print('Hello World!')
        return transformed_data + 1

    @graph(out={'one': GraphOut(), 'two': GraphOut()})
    def nested():
        if False:
            print('Hello World!')
        result = transform(return_stuff())
        return (one_more_transformation(result), result)

    @graph(out={'asset_1': GraphOut(), 'asset_2': GraphOut(), 'asset_3': GraphOut()})
    def complicated_graph():
        if False:
            while True:
                i = 10
        (one, two) = nested()
        return (one, two, transform(two))
    defs = Definitions(assets=[AssetsDefinition.from_graph(complicated_graph)], jobs=[define_asset_job('foo_job')])
    foo_job = defs.get_job_def('foo_job')
    result = foo_job.execute_in_process()
    assert result.success
    assert result.output_for_node('complicated_graph.nested', 'one') == 3
    assert result.output_for_node('complicated_graph.nested', 'two') == 2
    assert result.output_for_node('complicated_graph', 'asset_1') == 3
    assert result.output_for_node('complicated_graph', 'asset_2') == 2
    assert result.output_for_node('complicated_graph', 'asset_3') == 4

@ignore_warning('Parameter `io_manager_def` .* is experimental')
def test_source_asset_io_manager_def():
    if False:
        for i in range(10):
            print('nop')

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            pass

        def load_input(self, context):
            if False:
                return 10
            return 5

    @io_manager
    def the_manager():
        if False:
            i = 10
            return i + 15
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_def=the_manager)

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            while True:
                i = 10
        return my_source_asset + 4
    source_asset_job = build_assets_job(name='test', assets=[my_derived_asset], source_assets=[my_source_asset])
    result = source_asset_job.execute_in_process(asset_selection=[AssetKey('my_derived_asset')])
    assert result.success
    assert result.output_for_node('my_derived_asset') == 9

def test_source_asset_io_manager_not_provided():
    if False:
        while True:
            i = 10

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def load_input(self, context):
            if False:
                return 10
            return 5

    @io_manager
    def the_manager():
        if False:
            i = 10
            return i + 15
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'))

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            return 10
        return my_source_asset + 4
    source_asset_job = build_assets_job('the_job', assets=[my_derived_asset], source_assets=[my_source_asset], resource_defs={'io_manager': the_manager})
    result = source_asset_job.execute_in_process(asset_selection=[AssetKey('my_derived_asset')])
    assert result.success
    assert result.output_for_node('my_derived_asset') == 9

def test_source_asset_io_manager_key_provided():
    if False:
        print('Hello World!')

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            pass

        def load_input(self, context):
            if False:
                for i in range(10):
                    print('nop')
            return 5

    @io_manager
    def the_manager():
        if False:
            print('Hello World!')
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_key='some_key')

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            i = 10
            return i + 15
        return my_source_asset + 4
    source_asset_job = build_assets_job('the_job', assets=[my_derived_asset], source_assets=[my_source_asset], resource_defs={'some_key': the_manager})
    result = source_asset_job.execute_in_process(asset_selection=[AssetKey('my_derived_asset')])
    assert result.success
    assert result.output_for_node('my_derived_asset') == 9

@ignore_warning('Parameter `resource_defs` .* is experimental')
@ignore_warning('Parameter `io_manager_def` .* is experimental')
def test_source_asset_requires_resource_defs():
    if False:
        i = 10
        return i + 15

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            pass

        def load_input(self, context):
            if False:
                print('Hello World!')
            return 5

    @resource(required_resource_keys={'bar'})
    def foo_resource(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.bar == 'blah'

    @io_manager(required_resource_keys={'foo'})
    def the_manager():
        if False:
            while True:
                i = 10
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_def=the_manager, resource_defs={'foo': foo_resource, 'bar': ResourceDefinition.hardcoded_resource('blah')})

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            while True:
                i = 10
        return my_source_asset + 4
    source_asset_job = build_assets_job('the_job', assets=[my_derived_asset], source_assets=[my_source_asset])
    result = source_asset_job.execute_in_process(asset_selection=[AssetKey('my_derived_asset')])
    assert result.success
    assert result.output_for_node('my_derived_asset') == 9

@ignore_warning('Parameter `resource_defs` .* is experimental')
def test_other_asset_provides_req():
    if False:
        return 10

    @asset(required_resource_keys={'foo'})
    def asset_reqs_foo():
        if False:
            for i in range(10):
                print('nop')
        pass

    @asset(resource_defs={'foo': ResourceDefinition.hardcoded_resource('blah')})
    def asset_provides_foo():
        if False:
            return 10
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by op 'asset_reqs_foo' was not provided."):
        build_assets_job(name='test', assets=[asset_reqs_foo, asset_provides_foo])

@ignore_warning('Parameter `resource_defs` .* is experimental')
def test_transitive_deps_not_provided():
    if False:
        i = 10
        return i + 15

    @resource(required_resource_keys={'foo'})
    def unused_resource():
        if False:
            return 10
        pass

    @asset(resource_defs={'unused': unused_resource})
    def the_asset():
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by resource with key 'unused' was not provided."):
        build_assets_job(name='test', assets=[the_asset])

@ignore_warning('Parameter `resource_defs` .* is experimental')
def test_transitive_resource_deps_provided():
    if False:
        print('Hello World!')

    @resource(required_resource_keys={'foo'})
    def used_resource(context):
        if False:
            print('Hello World!')
        assert context.resources.foo == 'blah'

    @asset(resource_defs={'used': used_resource, 'foo': ResourceDefinition.hardcoded_resource('blah')})
    def the_asset():
        if False:
            return 10
        pass
    the_job = build_assets_job(name='test', assets=[the_asset])
    assert the_job.execute_in_process().success

@ignore_warning('Parameter `io_manager_def` .* is experimental')
def test_transitive_io_manager_dep_not_provided():
    if False:
        while True:
            i = 10

    @io_manager(required_resource_keys={'foo'})
    def the_manager():
        if False:
            for i in range(10):
                print('nop')
        pass
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_def=the_manager)

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by resource with key 'my_source_asset__io_manager' was not provided."):
        build_assets_job(name='test', assets=[my_derived_asset], source_assets=[my_source_asset])

def test_resolve_dependency_in_group():
    if False:
        i = 10
        return i + 15

    @asset(key_prefix='abc')
    def asset1():
        if False:
            while True:
                i = 10
        ...

    @asset
    def asset2(context, asset1):
        if False:
            return 10
        del asset1
        assert context.asset_key_for_input('asset1') == AssetKey(['abc', 'asset1'])
    with disable_dagster_warnings():
        assert materialize_to_memory([asset1, asset2]).success

def test_resolve_dependency_fail_across_groups():
    if False:
        i = 10
        return i + 15

    @asset(key_prefix='abc', group_name='other')
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        ...

    @asset
    def asset2(asset1):
        if False:
            i = 10
            return i + 15
        del asset1
    with pytest.raises(DagsterInvalidDefinitionError, match='is not produced by any of the provided asset ops and is not one of the provided sources'):
        with disable_dagster_warnings():
            materialize_to_memory([asset1, asset2])

def test_resolve_dependency_multi_asset_different_groups():
    if False:
        i = 10
        return i + 15

    @asset(key_prefix='abc', group_name='a')
    def upstream():
        if False:
            return 10
        ...

    @op(out={'ns1': Out(), 'ns2': Out()})
    def op1(upstream):
        if False:
            for i in range(10):
                print('nop')
        del upstream
    assets = AssetsDefinition(keys_by_input_name={'upstream': AssetKey('upstream')}, keys_by_output_name={'ns1': AssetKey('ns1'), 'ns2': AssetKey('ns2')}, node_def=op1, group_names_by_key={AssetKey('ns1'): 'a', AssetKey('ns2'): 'b'})
    with pytest.raises(DagsterInvalidDefinitionError, match='is not produced by any of the provided asset ops and is not one of the provided sources'):
        with disable_dagster_warnings():
            materialize_to_memory([upstream, assets])

def test_get_base_asset_jobs_multiple_partitions_defs():
    if False:
        while True:
            i = 10

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2021-05-05'))
    def daily_asset():
        if False:
            print('Hello World!')
        ...

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2021-05-05'))
    def daily_asset2():
        if False:
            print('Hello World!')
        ...

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2020-05-05'))
    def daily_asset_different_start_date():
        if False:
            print('Hello World!')
        ...

    @asset(partitions_def=HourlyPartitionsDefinition(start_date='2021-05-05-00:00'))
    def hourly_asset():
        if False:
            print('Hello World!')
        ...

    @asset
    def unpartitioned_asset():
        if False:
            i = 10
            return i + 15
        ...
    jobs = get_base_asset_jobs(assets=[daily_asset, daily_asset2, daily_asset_different_start_date, hourly_asset, unpartitioned_asset], source_assets=[], executor_def=None, resource_defs={}, asset_checks=[])
    assert len(jobs) == 3
    assert {job_def.name for job_def in jobs} == {'__ASSET_JOB_0', '__ASSET_JOB_1', '__ASSET_JOB_2'}
    assert {frozenset([node_def.name for node_def in job_def.all_node_defs]) for job_def in jobs} == {frozenset(['daily_asset', 'daily_asset2', 'unpartitioned_asset']), frozenset(['hourly_asset', 'unpartitioned_asset']), frozenset(['daily_asset_different_start_date', 'unpartitioned_asset'])}

@ignore_warning('Function `observable_source_asset` is experimental')
def test_get_base_asset_jobs_multiple_partitions_defs_and_observable_assets():
    if False:
        for i in range(10):
            print('nop')

    class B:
        ...
    partitions_a = StaticPartitionsDefinition(['a1'])

    @observable_source_asset(partitions_def=partitions_a)
    def asset_a():
        if False:
            while True:
                i = 10
        ...
    partitions_b = StaticPartitionsDefinition(['b1'])

    @observable_source_asset(partitions_def=partitions_b)
    def asset_b():
        if False:
            print('Hello World!')
        ...

    @asset(partitions_def=partitions_b)
    def asset_x(asset_b: B):
        if False:
            i = 10
            return i + 15
        ...
    jobs = get_base_asset_jobs(assets=[asset_x], source_assets=[asset_a, asset_b], executor_def=None, resource_defs={}, asset_checks=[])
    assert len(jobs) == 2
    assert {job_def.name for job_def in jobs} == {'__ASSET_JOB_0', '__ASSET_JOB_1'}

def test_coerce_resource_build_asset_job() -> None:
    if False:
        return 10
    executed = {}

    class BareResourceObject:
        pass

    @asset(required_resource_keys={'bare_resource'})
    def an_asset(context) -> None:
        if False:
            while True:
                i = 10
        assert context.resources.bare_resource
        executed['yes'] = True
    a_job = build_assets_job('my_job', assets=[an_asset], resource_defs={'bare_resource': BareResourceObject()})
    assert a_job.execute_in_process().success

def test_assets_def_takes_bare_object():
    if False:
        while True:
            i = 10

    class BareResourceObject:
        pass
    executed = {}

    @op(required_resource_keys={'bare_resource'})
    def an_op(context):
        if False:
            i = 10
            return i + 15
        assert context.resources.bare_resource
        executed['yes'] = True
    cool_thing_asset = AssetsDefinition(keys_by_input_name={}, keys_by_output_name={'result': AssetKey('thing')}, node_def=an_op, resource_defs={'bare_resource': BareResourceObject()})
    job = build_assets_job('graph_asset_job', [cool_thing_asset])
    result = job.execute_in_process()
    assert result.success
    assert executed['yes']

def test_async_multi_asset():
    if False:
        for i in range(10):
            print('nop')

    async def make_outputs():
        yield Output(1, output_name='A')
        yield Output(2, output_name='B')

    @multi_asset(outs={'A': AssetOut(), 'B': AssetOut()}, can_subset=True)
    async def aio_gen_asset(context):
        async for v in make_outputs():
            context.log.info(v.output_name)
            yield v
    aio_job = build_assets_job(name='test', assets=[aio_gen_asset])
    result = aio_job.execute_in_process()
    assert result.success

def test_selection_multi_component():
    if False:
        for i in range(10):
            print('nop')
    source_asset = SourceAsset(['apple', 'banana'])

    @asset(key_prefix='abc')
    def asset1():
        if False:
            i = 10
            return i + 15
        ...
    assert Definitions(assets=[source_asset, asset1], jobs=[define_asset_job('something', selection='abc/asset1')]).get_job_def('something').asset_layer.asset_keys == {AssetKey(['abc', 'asset1'])}

@pytest.mark.parametrize('job_selection,expected_nodes', [('*', 'n1,n2,n3'), ('n2+', 'n2,n3'), ('n1', 'n1')])
def test_asset_subset_io_managers(job_selection, expected_nodes):
    if False:
        while True:
            i = 10

    @io_manager(config_schema={'n': int})
    def return_n_io_manager(context):
        if False:
            while True:
                i = 10

        class ReturnNIOManager(IOManager):

            def handle_output(self, _context, obj):
                if False:
                    return 10
                pass

            def load_input(self, _context):
                if False:
                    return 10
                return context.resource_config['n']
        return ReturnNIOManager()
    _ACTUAL_OUTPUT_VAL = 99999

    @asset(io_manager_key='n1_iom')
    def n1():
        if False:
            print('Hello World!')
        return _ACTUAL_OUTPUT_VAL

    @asset(io_manager_key='n2_iom')
    def n2(n1):
        if False:
            i = 10
            return i + 15
        assert n1 == 1
        return _ACTUAL_OUTPUT_VAL

    @asset(io_manager_key='n3_iom')
    def n3(n1, n2):
        if False:
            print('Hello World!')
        assert n1 == 1
        assert n2 == 2
        return _ACTUAL_OUTPUT_VAL
    asset_job = define_asset_job('test', selection=job_selection)
    defs = Definitions(assets=[n1, n2, n3], resources={'n1_iom': return_n_io_manager.configured({'n': 1}), 'n2_iom': return_n_io_manager.configured({'n': 2}), 'n3_iom': return_n_io_manager.configured({'n': 3})}, jobs=[asset_job])
    result = defs.get_job_def('test').execute_in_process()
    for node in expected_nodes.split(','):
        assert result.output_for_node(node) == _ACTUAL_OUTPUT_VAL

def asset_aware_io_manager():
    if False:
        i = 10
        return i + 15

    class MyIOManager(IOManager):

        def __init__(self):
            if False:
                return 10
            self.db = {}

        def handle_output(self, context, obj):
            if False:
                print('Hello World!')
            self.db[context.asset_key] = obj

        def load_input(self, context):
            if False:
                return 10
            return self.db.get(context.asset_key)
    io_manager_obj = MyIOManager()

    @io_manager
    def _asset_aware():
        if False:
            i = 10
            return i + 15
        return io_manager_obj
    return (io_manager_obj, _asset_aware)

def _get_assets_defs(use_multi: bool=False, allow_subset: bool=False):
    if False:
        return 10
    'Get a predefined set of assets definitions for testing.\n\n    Dependencies:\n        "upstream": {\n            "start": set(),\n            "a": {"start"},\n            "b": set(),\n            "c": {"b"},\n            "d": {"a", "b"},\n            "e": {"c"},\n            "f": {"e", "d"},\n            "final": {"a", "d"},\n        },\n        "downstream": {\n            "start": {"a"},\n            "b": {"c", "d"},\n            "a": {"final", "d"},\n            "c": {"e"},\n            "d": {"final", "f"},\n            "e": {"f"},\n        }\n    '

    @asset
    def start():
        if False:
            while True:
                i = 10
        return 1

    @asset
    def a(start):
        if False:
            print('Hello World!')
        return start + 1

    @asset
    def b():
        if False:
            while True:
                i = 10
        return 1

    @asset
    def c(b):
        if False:
            for i in range(10):
                print('nop')
        return b + 1

    @multi_asset(outs={'a': AssetOut(is_required=False), 'b': AssetOut(is_required=False), 'c': AssetOut(is_required=False)}, internal_asset_deps={'a': {AssetKey('start')}, 'b': set(), 'c': {AssetKey('b')}}, can_subset=allow_subset)
    def abc_(context, start):
        if False:
            for i in range(10):
                print('nop')
        assert (context.selected_output_names != {'a', 'b', 'c'}) == context.is_subset
        a = start + 1 if start else None
        b = 1
        c = b + 1
        out_values = {'a': a, 'b': b, 'c': c}
        outputs_to_return = sorted(context.selected_output_names) if allow_subset else 'abc'
        for output_name in outputs_to_return:
            yield Output(out_values[output_name], output_name)

    @asset
    def d(a, b):
        if False:
            print('Hello World!')
        return a + b

    @asset
    def e(c):
        if False:
            print('Hello World!')
        return c + 1

    @asset
    def f(d, e):
        if False:
            i = 10
            return i + 15
        return d + e

    @multi_asset(outs={'d': AssetOut(is_required=False), 'e': AssetOut(is_required=False), 'f': AssetOut(is_required=False)}, internal_asset_deps={'d': {AssetKey('a'), AssetKey('b')}, 'e': {AssetKey('c')}, 'f': {AssetKey('d'), AssetKey('e')}}, can_subset=allow_subset)
    def def_(context, a, b, c):
        if False:
            print('Hello World!')
        assert (context.selected_output_names != {'d', 'e', 'f'}) == context.is_subset
        d = a + b if a and b else None
        e = c + 1 if c else None
        f = d + e if d and e else None
        out_values = {'d': d, 'e': e, 'f': f}
        outputs_to_return = sorted(context.selected_output_names) if allow_subset else 'def'
        for output_name in outputs_to_return:
            yield Output(out_values[output_name], output_name)

    @asset
    def final(a, d):
        if False:
            return 10
        return a + d
    if use_multi:
        return [start, abc_, def_, final]
    return [start, a, b, c, d, e, f, final]

@pytest.mark.parametrize('job_selection,use_multi,expected_error', [('*', False, None), ('*', True, None), ('e', False, None), ('e', True, (DagsterInvalidSubsetError, '')), ('x', False, (DagsterInvalidSubsetError, 'no AssetsDefinition objects supply these keys')), ('x', True, (DagsterInvalidSubsetError, 'no AssetsDefinition objects supply these keys')), (['start', 'x'], False, (DagsterInvalidSubsetError, 'no AssetsDefinition objects supply these keys')), (['start', 'x'], True, (DagsterInvalidSubsetError, 'no AssetsDefinition objects supply these keys')), (['d', 'e', 'f'], False, None), (['d', 'e', 'f'], True, None), (['start+'], False, None), (['start+'], True, (DagsterInvalidSubsetError, "When building job, the AssetsDefinition 'abc_' contains asset keys \\[AssetKey\\(\\['a'\\]\\), AssetKey\\(\\['b'\\]\\), AssetKey\\(\\['c'\\]\\)\\] and check keys \\[\\], but attempted to select only \\[AssetKey\\(\\['a'\\]\\)\\]"))])
def test_build_subset_job_errors(job_selection, use_multi, expected_error):
    if False:
        for i in range(10):
            print('nop')
    assets = _get_assets_defs(use_multi=use_multi)
    asset_job = define_asset_job('some_name', selection=job_selection)
    if expected_error:
        (expected_class, expected_message) = expected_error
        with pytest.raises(expected_class, match=expected_message):
            Definitions(assets=assets, jobs=[asset_job])
    else:
        Definitions(assets=assets, jobs=[asset_job])

def test_subset_does_not_respect_context():
    if False:
        print('Hello World!')

    @asset
    def start():
        if False:
            while True:
                i = 10
        return 1

    @multi_asset(outs={'a': AssetOut(), 'b': AssetOut(), 'c': AssetOut()}, can_subset=True)
    def abc(start):
        if False:
            print('Hello World!')
        yield Output(1 + start, 'a')
        yield Output(2 + start, 'b')
        yield Output(3 + start, 'c')

    @asset
    def final(c):
        if False:
            return 10
        return c + 1
    defs = Definitions([start, abc, final], jobs=[define_asset_job('subset_job', selection=['*final'])])
    job = defs.get_job_def('subset_job')
    specified_keys = {AssetKey('start'), AssetKey('c'), AssetKey('final')}
    with instance_for_test() as instance:
        result = job.execute_in_process(instance=instance)
        planned_asset_keys = {record.event_log_entry.dagster_event.event_specific_data.asset_key for record in instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION_PLANNED))}
    assert planned_asset_keys == specified_keys
    assert _all_asset_keys(result) == specified_keys | {AssetKey('a'), AssetKey('b')}

@pytest.mark.parametrize('job_selection,expected_assets', [('*', 'a,b,c'), ('a+', 'a,b'), ('+c', 'b,c'), (['a', 'c'], 'a,c')])
def test_simple_graph_backed_asset_subset(job_selection: CoercibleToAssetSelection, expected_assets: str):
    if False:
        return 10

    @op
    def one():
        if False:
            print('Hello World!')
        return 1

    @op
    def add_one(x):
        if False:
            i = 10
            return i + 15
        return x + 1

    @op(out=Out(io_manager_key='asset_io_manager'))
    def create_asset(x):
        if False:
            i = 10
            return i + 15
        return x * 2

    @graph
    def a():
        if False:
            print('Hello World!')
        return create_asset(add_one(add_one(one())))

    @graph
    def b(a):
        if False:
            print('Hello World!')
        return create_asset(add_one(add_one(a)))

    @graph
    def c(b):
        if False:
            i = 10
            return i + 15
        return create_asset(add_one(add_one(b)))
    a_asset = AssetsDefinition.from_graph(a)
    b_asset = AssetsDefinition.from_graph(b)
    c_asset = AssetsDefinition.from_graph(c)
    (_, io_manager_def) = asset_aware_io_manager()
    defs = Definitions(assets=[a_asset, b_asset, c_asset], jobs=[define_asset_job('assets_job', job_selection)], resources={'asset_io_manager': io_manager_def})
    defs.get_implicit_global_asset_job_def().execute_in_process()
    job = defs.get_job_def('assets_job')
    result = job.execute_in_process()
    expected_asset_keys = set((AssetKey(a) for a in expected_assets.split(',')))
    assert _all_asset_keys(result) == expected_asset_keys
    if AssetKey('a') in expected_asset_keys:
        assert result.output_for_node('a.create_asset') == 6
    if AssetKey('b') in expected_asset_keys:
        assert result.output_for_node('b.create_asset') == 16
    if AssetKey('c') in expected_asset_keys:
        assert result.output_for_node('c.create_asset') == 36

@pytest.mark.parametrize('use_multi', [True, False])
@pytest.mark.parametrize('job_selection,expected_assets,prefixes', [('*', 'start,a,b,c,d,e,f,final', None), ('a', 'a', None), ('b+', 'b,c,d', None), ('+f', 'f,d,e', None), ('++f', 'f,d,e,c,a,b', None), ('start*', 'start,a,d,f,final', None), (['+a', 'b+'], 'start,a,b,c,d', None), (['*c', 'final'], 'b,c,final', None), ('*', 'start,a,b,c,d,e,f,final', ['core', 'models']), ('core/models/a', 'a', ['core', 'models']), ('core/models/b+', 'b,c,d', ['core', 'models']), ('+core/models/f', 'f,d,e', ['core', 'models']), ('++core/models/f', 'f,d,e,c,a,b', ['core', 'models']), ('core/models/start*', 'start,a,d,f,final', ['core', 'models']), (['+core/models/a', 'core/models/b+'], 'start,a,b,c,d', ['core', 'models']), (['*core/models/c', 'core/models/final'], 'b,c,final', ['core', 'models'])])
def test_asset_group_build_subset_job(job_selection, expected_assets, use_multi, prefixes):
    if False:
        for i in range(10):
            print('nop')
    (_, io_manager_def) = asset_aware_io_manager()
    all_assets = _get_assets_defs(use_multi=use_multi, allow_subset=use_multi)
    for prefix in reversed(prefixes or []):
        (all_assets, _) = prefix_assets(all_assets, prefix, [], None)
    defs = Definitions(assets=all_assets, jobs=[define_asset_job('assets_job', job_selection)], resources={'io_manager': io_manager_def})
    defs.get_implicit_global_asset_job_def().execute_in_process()
    job = defs.get_job_def('assets_job')
    with instance_for_test() as instance:
        result = job.execute_in_process(instance=instance)
        planned_asset_keys = {record.event_log_entry.dagster_event.event_specific_data.asset_key for record in instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION_PLANNED))}
    expected_asset_keys = set((AssetKey([*(prefixes or []), a]) for a in expected_assets.split(',')))
    assert planned_asset_keys == expected_asset_keys
    assert _all_asset_keys(result) == expected_asset_keys
    if use_multi:
        expected_outputs = {'start': 1, 'abc_.a': 2, 'abc_.b': 1, 'abc_.c': 2, 'def_.d': 3, 'def_.e': 3, 'def_.f': 6, 'final': 5}
    else:
        expected_outputs = {'start': 1, 'a': 2, 'b': 1, 'c': 2, 'd': 3, 'e': 3, 'f': 6, 'final': 5}
    for (output, value) in expected_outputs.items():
        asset_name = output.split('.')[-1]
        if asset_name in expected_assets.split(','):
            if output != asset_name:
                node_def_name = output.split('.')[0]
                keys_for_node = {AssetKey([*(prefixes or []), c]) for c in node_def_name[:-1]}
                selected_keys_for_node = keys_for_node.intersection(expected_asset_keys)
                if selected_keys_for_node != keys_for_node and (not result.job_def.has_node_named(node_def_name)):
                    node_def_name += '_subset_' + hashlib.md5(str(list(sorted(selected_keys_for_node))).encode()).hexdigest()[-5:]
                assert result.output_for_node(node_def_name, asset_name)
            else:
                assert result.output_for_node(output, 'result') == value

def test_subset_cycle_resolution_embed_assets_in_complex_graph():
    if False:
        return 10
    'This represents a single large multi-asset with two assets embedded inside of it.\n\n    Ops:\n        foo produces: a, b, c, d, e, f, g, h\n        x produces: x\n        y produces: y\n\n    Upstream Assets:\n        a: []\n        b: []\n        c: [b]\n        d: [b]\n        e: [x, c]\n        f: [d]\n        g: [e]\n        h: [g, y]\n        x: [a]\n        y: [e, f].\n    '
    (io_manager_obj, io_manager_def) = asset_aware_io_manager()
    for item in 'abcdefghxy':
        io_manager_obj.db[AssetKey(item)] = None

    @multi_asset(outs={name: AssetOut(is_required=False) for name in 'a,b,c,d,e,f,g,h'.split(',')}, internal_asset_deps={'a': set(), 'b': set(), 'c': {AssetKey('b')}, 'd': {AssetKey('b')}, 'e': {AssetKey('c'), AssetKey('x')}, 'f': {AssetKey('d')}, 'g': {AssetKey('e')}, 'h': {AssetKey('g'), AssetKey('y')}}, can_subset=True)
    def foo(context, x, y):
        if False:
            return 10
        assert (context.selected_output_names != {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}) == context.is_subset
        a = b = c = d = e = f = g = h = None
        if 'a' in context.selected_output_names:
            a = 1
            yield Output(a, 'a')
        if 'b' in context.selected_output_names:
            b = 1
            yield Output(b, 'b')
        if 'c' in context.selected_output_names:
            c = (b or 1) + 1
            yield Output(c, 'c')
        if 'd' in context.selected_output_names:
            d = (b or 1) + 1
            yield Output(d, 'd')
        if 'e' in context.selected_output_names:
            e = x + (c or 2)
            yield Output(e, 'e')
        if 'f' in context.selected_output_names:
            f = (d or 1) + 1
            yield Output(f, 'f')
        if 'g' in context.selected_output_names:
            g = (e or 4) + 1
            yield Output(g, 'g')
        if 'h' in context.selected_output_names:
            h = (g or 5) + y
            yield Output(h, 'h')

    @asset
    def x(a):
        if False:
            print('Hello World!')
        return a + 1

    @asset
    def y(e, f):
        if False:
            for i in range(10):
                print('nop')
        return e + f
    job = Definitions(assets=[foo, x, y], resources={'io_manager': io_manager_def}).get_implicit_global_asset_job_def()
    assert len(list(job.graph.iterate_op_defs())) == 5
    result = job.execute_in_process()
    assert _all_asset_keys(result) == {AssetKey(x) for x in 'a,b,c,d,e,f,g,h,x,y'.split(',')}
    assert result.output_for_node('foo_3', 'h') == 12

def test_subset_cycle_resolution_complex():
    if False:
        for i in range(10):
            print('nop')
    'Test cycle resolution.\n\n    Ops:\n        foo produces: a, b, c, d, e, f\n        x produces: x\n        y produces: y\n        z produces: z\n\n    Upstream Assets:\n        a: []\n        b: [x]\n        c: [x]\n        d: [y]\n        e: [c]\n        f: [d]\n        x: [a]\n        y: [b, c].\n    '
    (io_manager_obj, io_manager_def) = asset_aware_io_manager()
    for item in 'abcdefxy':
        io_manager_obj.db[AssetKey(item)] = None

    @multi_asset(outs={name: AssetOut(is_required=False) for name in 'a,b,c,d,e,f'.split(',')}, internal_asset_deps={'a': set(), 'b': {AssetKey('x')}, 'c': {AssetKey('x')}, 'd': {AssetKey('y')}, 'e': {AssetKey('c')}, 'f': {AssetKey('d')}}, can_subset=True)
    def foo(context, x, y):
        if False:
            print('Hello World!')
        if 'a' in context.selected_output_names:
            yield Output(1, 'a')
        if 'b' in context.selected_output_names:
            yield Output(x + 1, 'b')
        if 'c' in context.selected_output_names:
            c = x + 2
            yield Output(c, 'c')
        if 'd' in context.selected_output_names:
            d = y + 1
            yield Output(d, 'd')
        if 'e' in context.selected_output_names:
            yield Output(c + 1, 'e')
        if 'f' in context.selected_output_names:
            yield Output(d + 1, 'f')

    @asset
    def x(a):
        if False:
            i = 10
            return i + 15
        return a + 1

    @asset
    def y(b, c):
        if False:
            i = 10
            return i + 15
        return b + c
    job = Definitions(assets=[foo, x, y], resources={'io_manager': io_manager_def}).get_implicit_global_asset_job_def()
    assert len(list(job.graph.iterate_op_defs())) == 5
    result = job.execute_in_process()
    assert _all_asset_keys(result) == {AssetKey(x) for x in 'a,b,c,d,e,f,x,y'.split(',')}
    assert result.output_for_node('x') == 2
    assert result.output_for_node('y') == 7
    assert result.output_for_node('foo_3', 'f') == 9

def test_subset_cycle_resolution_basic():
    if False:
        print('Hello World!')
    "Ops:\n        foo produces: a, b\n        foo_prime produces: a', b'\n    Assets:\n        s -> a -> a' -> b -> b'.\n    "
    (io_manager_obj, io_manager_def) = asset_aware_io_manager()
    for item in 'a,b,a_prime,b_prime'.split(','):
        io_manager_obj.db[AssetKey(item)] = None
    io_manager_obj.db[AssetKey('s')] = 0
    s = SourceAsset('s')

    @multi_asset(outs={'a': AssetOut(is_required=False), 'b': AssetOut(is_required=False)}, internal_asset_deps={'a': {AssetKey('s')}, 'b': {AssetKey('a_prime')}}, can_subset=True)
    def foo(context, s, a_prime):
        if False:
            return 10
        context.log.info(context.selected_asset_keys)
        if AssetKey('a') in context.selected_asset_keys:
            yield Output(s + 1, 'a')
        if AssetKey('b') in context.selected_asset_keys:
            yield Output(a_prime + 1, 'b')

    @multi_asset(outs={'a_prime': AssetOut(is_required=False), 'b_prime': AssetOut(is_required=False)}, internal_asset_deps={'a_prime': {AssetKey('a')}, 'b_prime': {AssetKey('b')}}, can_subset=True)
    def foo_prime(context, a, b):
        if False:
            while True:
                i = 10
        context.log.info(context.selected_asset_keys)
        if AssetKey('a_prime') in context.selected_asset_keys:
            yield Output(a + 1, 'a_prime')
        if AssetKey('b_prime') in context.selected_asset_keys:
            yield Output(b + 1, 'b_prime')
    job = Definitions(assets=[foo, foo_prime, s], resources={'io_manager': io_manager_def}).get_implicit_global_asset_job_def()
    assert len(list(job.graph.iterate_op_defs())) == 4
    result = job.execute_in_process()
    assert result.output_for_node('foo', 'a') == 1
    assert result.output_for_node('foo_prime', 'a_prime') == 2
    assert result.output_for_node('foo_2', 'b') == 3
    assert result.output_for_node('foo_prime_2', 'b_prime') == 4
    assert _all_asset_keys(result) == {AssetKey('a'), AssetKey('b'), AssetKey('a_prime'), AssetKey('b_prime')}

def test_subset_cycle_dependencies():
    if False:
        i = 10
        return i + 15
    'Ops:\n        foo produces: top, a, b\n        python produces: python\n    Assets:\n        top -> python -> b\n        a -> b.\n    '
    (io_manager_obj, io_manager_def) = asset_aware_io_manager()
    for item in 'a,b,a_prime,b_prime'.split(','):
        io_manager_obj.db[AssetKey(item)] = None

    @multi_asset(outs={'top': AssetOut(is_required=False), 'a': AssetOut(is_required=False), 'b': AssetOut(is_required=False)}, ins={'python': AssetIn(dagster_type=Nothing)}, internal_asset_deps={'top': set(), 'a': set(), 'b': {AssetKey('a'), AssetKey('python')}}, can_subset=True)
    def foo(context):
        if False:
            return 10
        for output in ['top', 'a', 'b']:
            if output in context.selected_output_names:
                yield Output(output, output)

    @asset(deps=[AssetKey('top')])
    def python():
        if False:
            return 10
        return 1
    defs = Definitions(assets=[foo, python], resources={'io_manager': io_manager_def})
    job = defs.get_implicit_global_asset_job_def()
    assert len(list(job.graph.iterate_op_defs())) == 3
    assert job.graph.dependencies == {NodeInvocation(name='foo'): {}, NodeInvocation(name='foo', alias='foo_2'): {'__subset_input__a': DependencyDefinition(node='foo', output='a'), 'python': DependencyDefinition(node='python', output='result')}, NodeInvocation(name='python'): {'top': DependencyDefinition(node='foo', output='top')}}
    result = job.execute_in_process()
    assert result.success
    assert _all_asset_keys(result) == {AssetKey('a'), AssetKey('b'), AssetKey('top'), AssetKey('python')}
    job = job.get_subset(asset_selection={AssetKey('a'), AssetKey('b')})
    assert len(list(job.graph.iterate_op_defs())) == 2
    assert job.graph.dependencies == {NodeInvocation(name='foo'): {}, NodeInvocation(name='foo', alias='foo_2'): {'__subset_input__a': DependencyDefinition(node='foo', output='a')}}
    result = job.execute_in_process()
    assert result.success
    assert _all_asset_keys(result) == {AssetKey('a'), AssetKey('b')}