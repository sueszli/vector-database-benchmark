import pytest
from dagster import AssetKey, AssetOut, AssetsDefinition, DagsterInvalidConfigError, DagsterInvalidDefinitionError, DagsterInvariantViolationError, DailyPartitionsDefinition, GraphOut, IOManager, Out, Output, ResourceDefinition, SourceAsset, asset, graph, io_manager, materialize_to_memory, multi_asset, op, with_resources
from dagster._core.errors import DagsterInvalidInvocationError

def test_basic_materialize_to_memory():
    if False:
        while True:
            i = 10

    @asset
    def the_asset():
        if False:
            print('Hello World!')
        return 5
    result = materialize_to_memory([the_asset])
    assert result.success
    assert len(result.asset_materializations_for_node('the_asset')[0].metadata) == 0
    assert result.asset_value(the_asset.key) == 5

def test_materialize_config():
    if False:
        return 10

    @asset(config_schema={'foo_str': str})
    def the_asset_reqs_config(context):
        if False:
            while True:
                i = 10
        assert context.op_config['foo_str'] == 'foo'
    assert materialize_to_memory([the_asset_reqs_config], run_config={'ops': {'the_asset_reqs_config': {'config': {'foo_str': 'foo'}}}}).success

def test_materialize_bad_config():
    if False:
        while True:
            i = 10

    @asset(config_schema={'foo_str': str})
    def the_asset_reqs_config(context):
        if False:
            while True:
                i = 10
        assert context.op_config['foo_str'] == 'foo'
    with pytest.raises(DagsterInvalidConfigError, match='Error in config for job'):
        materialize_to_memory([the_asset_reqs_config], run_config={'ops': {'the_asset_reqs_config': {'config': {'bad': 'foo'}}}})

def test_materialize_resources():
    if False:
        i = 10
        return i + 15

    @asset(resource_defs={'foo': ResourceDefinition.hardcoded_resource('blah')})
    def the_asset(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.foo == 'blah'
    assert materialize_to_memory([the_asset]).success

def test_materialize_resource_instances():
    if False:
        print('Hello World!')

    @asset(required_resource_keys={'foo', 'bar'})
    def the_asset(context):
        if False:
            print('Hello World!')
        assert context.resources.foo == 'blah'
        assert context.resources.bar == 'baz'
    assert materialize_to_memory([the_asset], resources={'foo': ResourceDefinition.hardcoded_resource('blah'), 'bar': 'baz'}).success

def test_materialize_resources_not_satisfied():
    if False:
        while True:
            i = 10

    @asset(required_resource_keys={'foo'})
    def the_asset(context):
        if False:
            while True:
                i = 10
        assert context.resources.foo == 'blah'
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by op 'the_asset' was not provided"):
        materialize_to_memory([the_asset])
    with pytest.raises(DagsterInvariantViolationError, match="Attempted to call `materialize_to_memory` with a resource provided for io manager key 'io_manager'. Do not provide resources for io manager keys when calling `materialize_to_memory`, as it will override io management behavior for all keys."):
        materialize_to_memory(with_resources([the_asset], {'foo': ResourceDefinition.hardcoded_resource('blah')}))

def test_materialize_conflicting_resources():
    if False:
        while True:
            i = 10

    @asset(resource_defs={'foo': ResourceDefinition.hardcoded_resource('1')})
    def first():
        if False:
            for i in range(10):
                print('nop')
        pass

    @asset(resource_defs={'foo': ResourceDefinition.hardcoded_resource('2')})
    def second():
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="Conflicting versions of resource with key 'foo' were provided to different assets."):
        materialize_to_memory([first, second])
    with pytest.raises(DagsterInvalidInvocationError, match='AssetsDefinition with key \\["first"\\] has conflicting resource definitions with provided resources for the following keys: foo. Either remove the existing resources from the asset or change the resource keys'):
        materialize_to_memory([first], resources={'foo': ResourceDefinition.hardcoded_resource('2')})

def test_materialize_source_assets():
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
                while True:
                    i = 10
            return 5

    @io_manager
    def the_manager():
        if False:
            return 10
        return MyIOManager()
    the_source = SourceAsset(key=AssetKey(['the_source']), io_manager_def=the_manager)

    @asset
    def the_asset(the_source):
        if False:
            i = 10
            return i + 15
        return the_source + 1
    with pytest.raises(DagsterInvariantViolationError, match="Attempted to call `materialize_to_memory` with a resource provided for io manager key 'the_source__io_manager'. Do not provide resources for io manager keys when calling `materialize_to_memory`, as it will override io management behavior for all keys."):
        materialize_to_memory([the_asset, the_source])

def test_materialize_no_assets():
    if False:
        i = 10
        return i + 15
    assert materialize_to_memory([]).success

def test_materialize_graph_backed_asset():
    if False:
        i = 10
        return i + 15

    @asset
    def a():
        if False:
            return 10
        return 'a'

    @asset
    def b():
        if False:
            print('Hello World!')
        return 'b'

    @op
    def double_string(s):
        if False:
            i = 10
            return i + 15
        return s * 2

    @op
    def combine_strings(s1, s2):
        if False:
            return 10
        return s1 + s2

    @graph
    def create_cool_thing(a, b):
        if False:
            print('Hello World!')
        da = double_string(double_string(a))
        db = double_string(b)
        return combine_strings(da, db)
    cool_thing_asset = AssetsDefinition(keys_by_input_name={'a': AssetKey('a'), 'b': AssetKey('b')}, keys_by_output_name={'result': AssetKey('cool_thing')}, node_def=create_cool_thing)
    result = materialize_to_memory([cool_thing_asset, a, b])
    assert result.success
    assert result.output_for_node('create_cool_thing.combine_strings') == 'aaaabb'
    assert result.asset_value('cool_thing') == 'aaaabb'

def test_materialize_multi_asset():
    if False:
        return 10

    @op
    def upstream_op():
        if False:
            for i in range(10):
                print('nop')
        return 'foo'

    @op(out={'o1': Out(), 'o2': Out()})
    def two_outputs(upstream_op):
        if False:
            return 10
        o1 = upstream_op
        o2 = o1 + 'bar'
        return (o1, o2)

    @graph(out={'o1': GraphOut(), 'o2': GraphOut()})
    def thing():
        if False:
            while True:
                i = 10
        (o1, o2) = two_outputs(upstream_op())
        return (o1, o2)
    thing_asset = AssetsDefinition(keys_by_input_name={}, keys_by_output_name={'o1': AssetKey('thing'), 'o2': AssetKey('thing_2')}, node_def=thing, asset_deps={AssetKey('thing'): set(), AssetKey('thing_2'): {AssetKey('thing')}})

    @multi_asset(outs={'my_out_name': AssetOut(metadata={'foo': 'bar'}), 'my_other_out_name': AssetOut(metadata={'bar': 'foo'})}, internal_asset_deps={'my_out_name': {AssetKey('my_other_out_name')}, 'my_other_out_name': {AssetKey('thing')}})
    def multi_asset_with_internal_deps(thing):
        if False:
            return 10
        yield Output(2, 'my_other_out_name')
        yield Output(1, 'my_out_name')
    result = materialize_to_memory([thing_asset, multi_asset_with_internal_deps])
    assert result.success
    assert result.output_for_node('multi_asset_with_internal_deps', 'my_out_name') == 1
    assert result.output_for_node('multi_asset_with_internal_deps', 'my_other_out_name') == 2

def test_materialize_to_memory_partition_key():
    if False:
        for i in range(10):
            print('nop')

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2022-01-01'))
    def the_asset(context):
        if False:
            while True:
                i = 10
        assert context.asset_partition_key_for_output() == '2022-02-02'
    result = materialize_to_memory([the_asset], partition_key='2022-02-02')
    assert result.success

def test_materialize_tags():
    if False:
        while True:
            i = 10

    @asset
    def the_asset(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.get_tag('key1') == 'value1'
    result = materialize_to_memory([the_asset], tags={'key1': 'value1'})
    assert result.success
    assert result.dagster_run.tags == {'key1': 'value1'}

def test_materialize_to_memory_partition_key_and_run_config():
    if False:
        i = 10
        return i + 15

    @asset(config_schema={'value': str})
    def configurable(context):
        if False:
            return 10
        assert context.op_config['value'] == 'a'

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2022-09-11'))
    def partitioned(context):
        if False:
            while True:
                i = 10
        assert context.partition_key == '2022-09-11'
    assert materialize_to_memory([partitioned, configurable], partition_key='2022-09-11', run_config={'ops': {'configurable': {'config': {'value': 'a'}}}}).success

def test_materialize_to_memory_provided_io_manager_instance():
    if False:
        for i in range(10):
            print('nop')

    @io_manager
    def the_manager():
        if False:
            i = 10
            return i + 15
        pass

    @asset(io_manager_key='blah')
    def the_asset():
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvariantViolationError, match="Attempted to call `materialize_to_memory` with a resource provided for io manager key 'blah'. Do not provide resources for io manager keys when calling `materialize_to_memory`, as it will override io management behavior for all keys."):
        materialize_to_memory([the_asset], resources={'blah': the_manager})

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            pass

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            pass
    with pytest.raises(DagsterInvariantViolationError, match="Attempted to call `materialize_to_memory` with a resource provided for io manager key 'blah'. Do not provide resources for io manager keys when calling `materialize_to_memory`, as it will override io management behavior for all keys."):
        materialize_to_memory([the_asset], resources={'blah': MyIOManager()})

def test_raise_on_error():
    if False:
        while True:
            i = 10

    @asset
    def asset1():
        if False:
            print('Hello World!')
        raise ValueError()
    assert not materialize_to_memory([asset1], raise_on_error=False).success

def test_selection():
    if False:
        print('Hello World!')

    @asset
    def upstream():
        if False:
            i = 10
            return i + 15
        ...

    @asset
    def downstream(upstream):
        if False:
            while True:
                i = 10
        ...
    assets = [upstream, downstream]
    result1 = materialize_to_memory(assets, selection=[upstream])
    assert result1.success
    materialization_events = result1.get_asset_materialization_events()
    assert len(materialization_events) == 1
    assert materialization_events[0].materialization.asset_key == AssetKey('upstream')