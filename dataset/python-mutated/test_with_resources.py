import warnings
import pytest
from dagster import AssetKey, Definitions, IOManager, IOManagerDefinition, ResourceDefinition, build_op_context, execute_job, io_manager, mem_io_manager, reconstructable, resource
from dagster._core.definitions import AssetsDefinition, SourceAsset, asset, build_assets_job
from dagster._core.errors import DagsterInvalidConfigError, DagsterInvalidDefinitionError, DagsterInvalidInvocationError, DagsterInvariantViolationError
from dagster._core.execution.with_resources import with_resources
from dagster._core.storage.fs_io_manager import PickledObjectFilesystemIOManager
from dagster._core.storage.mem_io_manager import InMemoryIOManager
from dagster._core.test_utils import environ, instance_for_test

@pytest.fixture
def instance():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        yield instance

def test_assets_direct():
    if False:
        return 10

    @asset(required_resource_keys={'foo'})
    def the_asset(context):
        if False:
            return 10
        assert context.resources.foo == 'blah'
        return 5
    in_mem = InMemoryIOManager()

    @io_manager
    def the_io_manager():
        if False:
            for i in range(10):
                print('nop')
        return in_mem
    transformed_asset = with_resources([the_asset], {'foo': ResourceDefinition.hardcoded_resource('blah'), 'io_manager': the_io_manager})[0]
    assert isinstance(transformed_asset, AssetsDefinition)
    assert transformed_asset.node_def.output_defs[0].io_manager_key == 'io_manager'
    assert build_assets_job('the_job', [transformed_asset]).execute_in_process().success
    assert next(iter(in_mem.values.values())) == 5

def test_asset_requires_io_manager_key():
    if False:
        for i in range(10):
            print('nop')

    @asset(io_manager_key='the_manager')
    def the_asset():
        if False:
            i = 10
            return i + 15
        return 5
    in_mem = InMemoryIOManager()

    @io_manager
    def the_io_manager():
        if False:
            print('Hello World!')
        return in_mem
    transformed_asset = with_resources([the_asset], {'foo': ResourceDefinition.hardcoded_resource('blah'), 'the_manager': the_io_manager})[0]
    assert isinstance(transformed_asset, AssetsDefinition)
    assert build_assets_job('the_job', [transformed_asset]).execute_in_process().success
    assert next(iter(in_mem.values.values())) == 5

def test_assets_direct_resource_conflicts():
    if False:
        for i in range(10):
            print('nop')

    @asset(required_resource_keys={'foo'})
    def the_asset():
        if False:
            return 10
        pass

    @asset(required_resource_keys={'foo'})
    def other_asset():
        if False:
            print('Hello World!')
        pass
    transformed_asset = with_resources([the_asset], {'foo': ResourceDefinition.hardcoded_resource('blah')})[0]
    other_transformed_asset = with_resources([other_asset], {'foo': ResourceDefinition.hardcoded_resource('blah')})[0]
    with pytest.raises(DagsterInvalidDefinitionError, match="Conflicting versions of resource with key 'foo' were provided to different assets. When constructing a job, all resource definitions provided to assets must match by reference equality for a given key."):
        build_assets_job('the_job', [transformed_asset, other_transformed_asset])

def test_source_assets_no_key_provided():
    if False:
        i = 10
        return i + 15

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                print('Hello World!')
            pass

        def load_input(self, context):
            if False:
                return 10
            return 5

    @io_manager
    def the_manager():
        if False:
            for i in range(10):
                print('nop')
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'))

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            while True:
                i = 10
        return my_source_asset + 4
    (transformed_source, transformed_derived) = with_resources([my_source_asset, my_derived_asset], resource_defs={'io_manager': the_manager})
    assert transformed_source.get_io_manager_key() == 'io_manager'
    the_job = build_assets_job('the_job', [transformed_derived], source_assets=[transformed_source])
    result = the_job.execute_in_process()
    assert result.success
    assert result.output_for_node('my_derived_asset') == 9

def test_source_assets_key_provided():
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
                print('Hello World!')
            return 5

    @io_manager
    def the_manager():
        if False:
            while True:
                i = 10
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_key='the_manager')

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            return 10
        return my_source_asset + 4
    (transformed_source, transformed_derived) = with_resources([my_source_asset, my_derived_asset], resource_defs={'the_manager': the_manager})
    assert transformed_source.get_io_manager_key() == 'the_manager'
    the_job = build_assets_job('the_job', [transformed_derived], source_assets=[transformed_source])
    result = the_job.execute_in_process()
    assert result.success
    assert result.output_for_node('my_derived_asset') == 9

def test_source_assets_manager_def_provided():
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
                while True:
                    i = 10
            return 5

    @io_manager
    def the_manager():
        if False:
            print('Hello World!')
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_def=the_manager)

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            i = 10
            return i + 15
        return my_source_asset + 4
    (transformed_source, transformed_derived) = with_resources([my_source_asset, my_derived_asset], resource_defs={'io_manager': mem_io_manager})
    assert transformed_source.io_manager_def == the_manager
    the_job = build_assets_job('the_job', [transformed_derived], source_assets=[transformed_source])
    result = the_job.execute_in_process()
    assert result.success
    assert result.output_for_node('my_derived_asset') == 9

def test_asset_def_partial_application():
    if False:
        i = 10
        return i + 15

    @asset(required_resource_keys={'foo', 'bar'})
    def the_asset():
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'bar' required by op 'the_asset' was not provided."):
        with_resources([the_asset], {'foo': ResourceDefinition.hardcoded_resource('foo')})

def test_source_asset_no_manager_def():
    if False:
        print('Hello World!')
    the_source_asset = SourceAsset(key=AssetKey('my_source_asset'))
    with_resources([the_source_asset], {})
    the_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_key='foo')
    with pytest.raises(DagsterInvalidDefinitionError, match='io manager with key \'foo\' required by SourceAsset with key \\["my_source_asset"\\] was not provided'):
        with_resources([the_source_asset], {})

def test_asset_transitive_resource_deps():
    if False:
        print('Hello World!')

    @resource(required_resource_keys={'foo'})
    def the_resource(context):
        if False:
            while True:
                i = 10
        assert context.resources.foo == 'bar'

    @asset(resource_defs={'the_resource': the_resource})
    def the_asset():
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by resource with key 'the_resource' was not provided"):
        with_resources([the_asset], {})
    transformed_asset = with_resources([the_asset], {'foo': ResourceDefinition.hardcoded_resource('bar')})[0]
    assert build_assets_job('blah', [transformed_asset]).execute_in_process().success

def test_asset_io_manager_transitive_dependencies():
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
                i = 10
                return i + 15
            return 5

    @io_manager(required_resource_keys={'the_resource'})
    def the_manager():
        if False:
            i = 10
            return i + 15
        return MyIOManager()

    @asset(io_manager_def=the_manager)
    def the_asset():
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'the_resource' required by resource with key 'the_asset__io_manager' was not provided."):
        with_resources([the_asset], resource_defs={})

    @resource(required_resource_keys={'foo'})
    def the_resource(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.foo == 'bar'
    with pytest.raises(DagsterInvariantViolationError, match="Resource with key 'foo' required by resource with key 'the_resource', but not provided."):
        with_resources([the_asset], resource_defs={'the_resource': the_resource})
    transformed_assets = with_resources([the_asset], resource_defs={'the_resource': the_resource, 'foo': ResourceDefinition.hardcoded_resource('bar')})
    assert build_assets_job('blah', transformed_assets).execute_in_process().success

def test_source_asset_partial_resources():
    if False:
        i = 10
        return i + 15

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            pass

        def load_input(self, context):
            if False:
                for i in range(10):
                    print('nop')
            return 5

    @io_manager(required_resource_keys={'foo'})
    def the_manager(context):
        if False:
            i = 10
            return i + 15
        assert context.resources.foo == 'blah'
        return MyIOManager()
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_def=the_manager)
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by resource with key 'my_source_asset__io_manager' was not provided"):
        with_resources([my_source_asset], resource_defs={})

    @resource(required_resource_keys={'bar'})
    def foo_resource(context):
        if False:
            return 10
        return context.resources.bar
    with pytest.raises(DagsterInvariantViolationError, match="Resource with key 'bar' required by resource with key 'foo', but not provided."):
        with_resources([my_source_asset], resource_defs={'foo': foo_resource})
    transformed_source = with_resources([my_source_asset], resource_defs={'foo': foo_resource, 'bar': ResourceDefinition.hardcoded_resource('blah')})[0]

    @asset
    def my_derived_asset(my_source_asset):
        if False:
            return 10
        return my_source_asset + 4
    the_job = build_assets_job('the_job', [my_derived_asset], source_assets=[transformed_source])
    result = the_job.execute_in_process()
    assert result.success
    assert result.output_for_node('my_derived_asset') == 9

def test_asset_circular_resource_dependency():
    if False:
        i = 10
        return i + 15

    @asset(required_resource_keys={'foo'})
    def the_asset():
        if False:
            for i in range(10):
                print('nop')
        pass

    @resource(required_resource_keys={'bar'})
    def foo():
        if False:
            return 10
        pass

    @resource(required_resource_keys={'foo'})
    def bar():
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(DagsterInvariantViolationError, match='Resource key "bar" transitively depends on itself.'):
        with_resources([the_asset], resource_defs={'foo': foo, 'bar': bar})

def get_resource_and_asset_for_config_tests():
    if False:
        return 10

    @asset(required_resource_keys={'foo', 'bar'})
    def the_asset(context):
        if False:
            return 10
        assert context.resources.foo == 'blah'
        assert context.resources.bar == 'baz'

    @resource(config_schema=str)
    def the_resource(context):
        if False:
            i = 10
            return i + 15
        return context.resource_config
    return (the_asset, the_resource)

def test_config():
    if False:
        return 10
    (the_asset, the_resource) = get_resource_and_asset_for_config_tests()
    transformed_asset = with_resources([the_asset], resource_defs={'foo': the_resource, 'bar': the_resource}, resource_config_by_key={'foo': {'config': 'blah'}, 'bar': {'config': 'baz'}})[0]
    transformed_asset(build_op_context())

def test_config_not_satisfied():
    if False:
        return 10
    (the_asset, the_resource) = get_resource_and_asset_for_config_tests()
    transformed_asset = with_resources([the_asset], resource_defs={'foo': the_resource, 'bar': the_resource})[0]
    result = build_assets_job('test', [transformed_asset], config={'resources': {'foo': {'config': 'blah'}, 'bar': {'config': 'baz'}}}).execute_in_process()
    assert result.success

def test_bad_key_provided():
    if False:
        return 10
    (the_asset, the_resource) = get_resource_and_asset_for_config_tests()
    transformed_asset = with_resources([the_asset], resource_defs={'foo': the_resource, 'bar': the_resource}, resource_config_by_key={'foo': {'config': 'blah'}, 'bar': {'config': 'baz'}, 'bad': 'whatever'})[0]
    transformed_asset(build_op_context())

def test_bad_config_provided():
    if False:
        i = 10
        return i + 15
    (the_asset, the_resource) = get_resource_and_asset_for_config_tests()
    with pytest.raises(DagsterInvalidConfigError, match="Error when applying config for resource with key 'foo'"):
        with_resources([the_asset], resource_defs={'foo': the_resource, 'bar': the_resource}, resource_config_by_key={'foo': {'config': object()}})
    with pytest.raises(DagsterInvalidInvocationError, match="Error with config for resource key 'foo'"):
        with_resources([the_asset], resource_defs={'foo': the_resource, 'bar': the_resource}, resource_config_by_key={'foo': 'bad'})

def test_overlapping_io_manager_asset():
    if False:
        return 10

    @io_manager
    def the_io_manager():
        if False:
            i = 10
            return i + 15
        pass

    @asset(io_manager_def=the_io_manager)
    def the_asset():
        if False:
            i = 10
            return i + 15
        pass
    with_resources([the_asset], resource_defs={'io_manager': mem_io_manager})
    with pytest.raises(DagsterInvalidInvocationError, match='AssetsDefinition with key \\[\\"the_asset\\"\\] has conflicting resource definitions with provided resources for the following keys: the_asset__io_manager. Either remove the existing resources from the asset or change the resource keys so that they don\'t overlap.'):
        with_resources([the_asset], resource_defs={'the_asset__io_manager': mem_io_manager})

def test_overlapping_resources_asset():
    if False:
        return 10
    foo_resource = ResourceDefinition.hardcoded_resource('blah')

    @asset(resource_defs={'foo': foo_resource})
    def the_asset():
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(DagsterInvalidInvocationError, match='AssetsDefinition with key \\[\\"the_asset\\"\\] has conflicting resource definitions with provided resources for the following keys: foo. Either remove the existing resources from the asset or change the resource keys so that they don\'t overlap.'):
        with_resources([the_asset], resource_defs={'foo': foo_resource})
    with pytest.raises(DagsterInvalidInvocationError, match='AssetsDefinition with key \\[\\"the_asset\\"\\] has conflicting resource definitions with provided resources for the following keys: foo. Either remove the existing resources from the asset or change the resource keys so that they don\'t overlap.'):
        with_resources([the_asset], resource_defs={'foo': ResourceDefinition.hardcoded_resource('diff_ref')})

def test_overlapping_io_manager_source_asset():
    if False:
        while True:
            i = 10

    @io_manager
    def the_io_manager():
        if False:
            return 10
        pass
    the_asset = SourceAsset(key=AssetKey('the_asset'), io_manager_def=the_io_manager)
    with_resources([the_asset], resource_defs={'io_manager': mem_io_manager})
    with pytest.raises(DagsterInvalidInvocationError, match="SourceAsset with key AssetKey\\(\\['the_asset'\\]\\) has conflicting resource definitions with provided resources for the following keys: \\['the_asset__io_manager'\\]. Either remove the existing resources from the asset or change the resource keys so that they don't overlap."):
        with_resources([the_asset], resource_defs={'the_asset__io_manager': mem_io_manager})

def test_overlapping_resources_source_asset():
    if False:
        print('Hello World!')
    foo_resource = ResourceDefinition.hardcoded_resource('blah')

    @io_manager(required_resource_keys={'foo'})
    def the_io_manager():
        if False:
            while True:
                i = 10
        pass
    the_asset = SourceAsset(key=AssetKey('the_asset'), io_manager_def=the_io_manager, resource_defs={'foo': foo_resource})
    with pytest.raises(DagsterInvalidInvocationError, match="SourceAsset with key AssetKey\\(\\['the_asset'\\]\\) has conflicting resource definitions with provided resources for the following keys: \\['foo'\\]. Either remove the existing resources from the asset or change the resource keys so that they don't overlap."):
        with_resources([the_asset], resource_defs={'foo': foo_resource})
    with pytest.raises(DagsterInvalidInvocationError, match="SourceAsset with key AssetKey\\(\\['the_asset'\\]\\) has conflicting resource definitions with provided resources for the following keys: \\['foo'\\]. Either remove the existing resources from the asset or change the resource keys so that they don't overlap."):
        with_resources([the_asset], resource_defs={'foo': ResourceDefinition.hardcoded_resource('diff_ref')})

def test_with_resources_no_exp_warnings():
    if False:
        print('Hello World!')

    @asset(required_resource_keys={'foo'})
    def blah():
        if False:
            for i in range(10):
                print('nop')
        pass

    @io_manager
    def the_manager():
        if False:
            i = 10
            return i + 15
        pass
    my_source_asset = SourceAsset(key=AssetKey('my_source_asset'), io_manager_key='the_manager')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with_resources([blah, my_source_asset], {'foo': ResourceDefinition.hardcoded_resource('something'), 'the_manager': the_manager})

def test_bare_resource_on_with_resources():
    if False:
        while True:
            i = 10

    class BareObjectResource:
        pass
    executed = {}

    @asset(required_resource_keys={'bare_resource'})
    def blah(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.bare_resource
        executed['yes'] = True
    bound_assets = with_resources([blah], {'bare_resource': BareObjectResource()})
    defs = Definitions(assets=bound_assets)
    defs.get_implicit_global_asset_job_def().execute_in_process()
    assert executed['yes']

class FooIoManager(PickledObjectFilesystemIOManager):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(base_dir='/tmp/dagster/foo-io-manager')
io_manager_resource_fn = lambda _: FooIoManager()
foo_io_manager_def = IOManagerDefinition(resource_fn=io_manager_resource_fn, config_schema={})

def create_asset_job():
    if False:
        i = 10
        return i + 15

    @asset
    def my_derived_asset():
        if False:
            for i in range(10):
                print('nop')
        return 4
    return build_assets_job('the_job', with_resources([my_derived_asset], {}))

def test_source_asset_default_io_manager(instance):
    if False:
        while True:
            i = 10
    with environ({'DAGSTER_DEFAULT_IO_MANAGER_MODULE': 'dagster_tests.core_tests.resource_tests.test_with_resources', 'DAGSTER_DEFAULT_IO_MANAGER_ATTRIBUTE': 'foo_io_manager_def'}):
        assert execute_job(reconstructable(create_asset_job), instance).success
    with environ({'DAGSTER_DEFAULT_IO_MANAGER_MODULE': 'dagster_tests.core_tests.resource_tests.fake_file', 'DAGSTER_DEFAULT_IO_MANAGER_ATTRIBUTE': 'foo_io_manager_def'}):
        assert not execute_job(reconstructable(create_asset_job), instance).success