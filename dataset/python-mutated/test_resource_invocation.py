from contextlib import contextmanager
import pytest
from dagster import Field, Noneable, Selector, build_init_resource_context, resource
from dagster._core.errors import DagsterInvalidConfigError, DagsterInvalidDefinitionError, DagsterInvalidInvocationError

def test_resource_invocation_no_arg():
    if False:
        while True:
            i = 10

    @resource
    def basic_resource():
        if False:
            i = 10
            return i + 15
        return 5
    assert basic_resource() == 5

def test_resource_invocation_none_arg():
    if False:
        i = 10
        return i + 15

    @resource
    def basic_resource(_):
        if False:
            return 10
        return 5
    assert basic_resource(None) == 5
    with pytest.raises(DagsterInvalidInvocationError, match='Resource initialization function has context argument, but no context was provided when invoking.'):
        basic_resource()

    @resource
    def basic_resource_arb_context(arb_context):
        if False:
            while True:
                i = 10
        return 5
    assert basic_resource_arb_context(None) == 5
    assert basic_resource_arb_context(arb_context=None) == 5
    with pytest.raises(DagsterInvalidInvocationError, match="Resource initialization expected argument 'arb_context'."):
        assert basic_resource_arb_context(wrong_context=None) == 5

def test_resource_invocation_with_resources():
    if False:
        print('Hello World!')

    @resource(required_resource_keys={'foo'})
    def resource_reqs_resources(init_context):
        if False:
            for i in range(10):
                print('nop')
        return init_context.resources.foo
    with pytest.raises(DagsterInvalidInvocationError, match='Resource has required resources, but no context was provided.'):
        resource_reqs_resources(None)
    context = build_init_resource_context()
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required was not provided."):
        resource_reqs_resources(context)
    context = build_init_resource_context(resources={'foo': 'bar'})
    assert resource_reqs_resources(context) == 'bar'

def test_resource_invocation_with_cm_resource():
    if False:
        return 10
    teardown_log = []

    @resource
    @contextmanager
    def cm_resource(_):
        if False:
            print('Hello World!')
        try:
            yield 'foo'
        finally:
            teardown_log.append('collected')
    with cm_resource(None) as resource_val:
        assert resource_val == 'foo'
        assert not teardown_log
    assert teardown_log == ['collected']

def test_resource_invocation_with_config():
    if False:
        i = 10
        return i + 15

    @resource(config_schema={'foo': str})
    def resource_reqs_config(context):
        if False:
            return 10
        assert context.resource_config['foo'] == 'bar'
        return 5
    with pytest.raises(DagsterInvalidInvocationError, match='Resource has required config schema, but no context was provided.'):
        resource_reqs_config(None)
    context = build_init_resource_context()
    with pytest.raises(DagsterInvalidConfigError, match='Error in config for resource'):
        resource_reqs_config(context)
    with pytest.raises(DagsterInvalidConfigError, match='Error when applying config mapping for resource'):
        resource_reqs_config.configured({'foobar': 'bar'})(None)
    result = resource_reqs_config.configured({'foo': 'bar'})(None)
    assert result == 5
    result = resource_reqs_config(build_init_resource_context(config={'foo': 'bar'}))
    assert result == 5

def test_failing_resource():
    if False:
        while True:
            i = 10

    @resource
    def fails(_):
        if False:
            i = 10
            return i + 15
        raise Exception('Oh no!')
    with pytest.raises(Exception, match='Oh no!'):
        fails(None)

def test_resource_invocation_dict_config():
    if False:
        i = 10
        return i + 15

    @resource(config_schema=dict)
    def resource_requires_dict(context):
        if False:
            print('Hello World!')
        assert context.resource_config == {'foo': 'bar'}
        return context.resource_config
    assert resource_requires_dict(build_init_resource_context(config={'foo': 'bar'})) == {'foo': 'bar'}

    @resource(config_schema=Noneable(dict))
    def resource_noneable_dict(context):
        if False:
            for i in range(10):
                print('nop')
        return context.resource_config
    assert resource_noneable_dict(build_init_resource_context()) is None
    assert resource_noneable_dict(None) is None

def test_resource_invocation_default_config():
    if False:
        while True:
            i = 10

    @resource(config_schema={'foo': Field(str, is_required=False, default_value='bar')})
    def resource_requires_config(context):
        if False:
            i = 10
            return i + 15
        assert context.resource_config['foo'] == 'bar'
        return context.resource_config['foo']
    assert resource_requires_config(None) == 'bar'

    @resource(config_schema=Field(str, is_required=False, default_value='bar'))
    def resource_requires_config_val(context):
        if False:
            print('Hello World!')
        assert context.resource_config == 'bar'
        return context.resource_config
    assert resource_requires_config_val(None) == 'bar'

    @resource(config_schema={'foo': Field(str, is_required=False, default_value='bar'), 'baz': str})
    def resource_requires_config_partial(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.resource_config['foo'] == 'bar'
        assert context.resource_config['baz'] == 'bar'
        return context.resource_config['foo'] + context.resource_config['baz']
    assert resource_requires_config_partial(build_init_resource_context(config={'baz': 'bar'})) == 'barbar'

def test_resource_invocation_kitchen_sink_config():
    if False:
        while True:
            i = 10

    @resource(config_schema={'str_field': str, 'int_field': int, 'list_int': [int], 'list_list_int': [[int]], 'dict_field': {'a_string': str}, 'list_dict_field': [{'an_int': int}], 'selector_of_things': Selector({'select_list_dict_field': [{'an_int': int}], 'select_int': int}), 'optional_list_of_optional_string': Noneable([Noneable(str)])})
    def kitchen_sink(context):
        if False:
            print('Hello World!')
        return context.resource_config
    resource_config = {'str_field': 'kjf', 'int_field': 2, 'list_int': [3], 'list_list_int': [[1], [2, 3]], 'dict_field': {'a_string': 'kdjfkd'}, 'list_dict_field': [{'an_int': 2}, {'an_int': 4}], 'selector_of_things': {'select_int': 3}, 'optional_list_of_optional_string': ['foo', None]}
    assert kitchen_sink(build_init_resource_context(config=resource_config)) == resource_config

def test_resource_dep_no_context():
    if False:
        print('Hello World!')

    @resource(required_resource_keys={'foo'})
    def the_resource():
        if False:
            for i in range(10):
                print('nop')
        pass
    the_resource()
    with pytest.raises(DagsterInvalidInvocationError, match='Attempted to invoke resource with argument, but underlying function has no context argument. Either specify a context argument on the resource function, or remove the passed-in argument.'):
        the_resource(None)