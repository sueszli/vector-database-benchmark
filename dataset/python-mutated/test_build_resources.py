import pytest
from dagster import Field, resource
from dagster._core.definitions.resource_definition import IContainsGenerator
from dagster._core.errors import DagsterResourceFunctionError
from dagster._core.execution.build_resources import build_resources

def test_basic_resource():
    if False:
        i = 10
        return i + 15

    @resource
    def basic_resource(_):
        if False:
            for i in range(10):
                print('nop')
        return 'foo'
    with build_resources(resources={'basic_resource': basic_resource}) as resources:
        assert not isinstance(resources, IContainsGenerator)
        assert resources.basic_resource == 'foo'

def test_resource_with_config():
    if False:
        for i in range(10):
            print('nop')

    @resource(config_schema={'plant': str, 'animal': Field(str, is_required=False, default_value='dog')})
    def basic_resource(init_context):
        if False:
            return 10
        plant = init_context.resource_config['plant']
        animal = init_context.resource_config['animal']
        return f'plant: {plant}, animal: {animal}'
    with build_resources(resources={'basic_resource': basic_resource}, resource_config={'basic_resource': {'config': {'plant': 'maple tree'}}}) as resources:
        assert resources.basic_resource == 'plant: maple tree, animal: dog'

def test_resource_with_dependencies():
    if False:
        for i in range(10):
            print('nop')

    @resource(config_schema={'animal': str})
    def no_deps(init_context):
        if False:
            print('Hello World!')
        return init_context.resource_config['animal']

    @resource(required_resource_keys={'no_deps'})
    def has_deps(init_context):
        if False:
            i = 10
            return i + 15
        return f'{init_context.resources.no_deps} is an animal.'
    with build_resources(resources={'no_deps': no_deps, 'has_deps': has_deps}, resource_config={'no_deps': {'config': {'animal': 'dog'}}}) as resources:
        assert resources.no_deps == 'dog'
        assert resources.has_deps == 'dog is an animal.'

def test_error_in_resource_initialization():
    if False:
        return 10

    @resource
    def i_will_fail(_):
        if False:
            i = 10
            return i + 15
        raise Exception('Failed.')
    with pytest.raises(DagsterResourceFunctionError, match='Error executing resource_fn on ResourceDefinition i_will_fail'):
        with build_resources(resources={'i_will_fail': i_will_fail}):
            pass

def test_resource_init_requires_instance():
    if False:
        for i in range(10):
            print('nop')

    @resource
    def basic_resource(init_context):
        if False:
            while True:
                i = 10
        assert init_context.instance
    build_resources({'basic_resource': basic_resource})

def test_resource_init_values():
    if False:
        i = 10
        return i + 15

    @resource(required_resource_keys={'bar'})
    def foo_resource(init_context):
        if False:
            print('Hello World!')
        assert init_context.resources.bar == 'bar'
        return 'foo'
    with build_resources({'foo': foo_resource, 'bar': 'bar'}) as resources:
        assert resources.foo == 'foo'
        assert resources.bar == 'bar'

def test_context_manager_resource():
    if False:
        for i in range(10):
            print('nop')
    tore_down = []

    @resource
    def cm_resource(_):
        if False:
            while True:
                i = 10
        try:
            yield 'foo'
        finally:
            tore_down.append('yes')
    with build_resources({'cm_resource': cm_resource}) as resources:
        assert isinstance(resources, IContainsGenerator)
        assert resources.cm_resource == 'foo'
    assert tore_down == ['yes']