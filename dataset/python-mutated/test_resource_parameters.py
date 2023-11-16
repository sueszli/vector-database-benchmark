import sys
from typing import Any
import pytest
from dagster import AssetsDefinition, ResourceDefinition, asset, job, op, resource, with_resources
from dagster._check import ParameterCheckError
from dagster._config.pythonic_config import Config
from dagster._core.definitions.asset_out import AssetOut
from dagster._core.definitions.assets_job import build_assets_job
from dagster._core.definitions.decorators.asset_decorator import multi_asset
from dagster._core.definitions.resource_annotation import ResourceParam
from dagster._core.errors import DagsterInvalidDefinitionError

def test_filter_out_resources():
    if False:
        return 10

    @op
    def requires_resource_a(context, a: ResourceParam[str]):
        if False:
            print('Hello World!')
        assert a
        assert context.resources.a
        assert not hasattr(context.resources, 'b')

    @op
    def requires_resource_b(context, b: ResourceParam[str]):
        if False:
            for i in range(10):
                print('nop')
        assert b
        assert not hasattr(context.resources, 'a')
        assert context.resources.b

    @op
    def not_resources(context):
        if False:
            i = 10
            return i + 15
        assert not hasattr(context.resources, 'a')
        assert not hasattr(context.resources, 'b')

    @job(resource_defs={'a': ResourceDefinition.hardcoded_resource('foo'), 'b': ResourceDefinition.hardcoded_resource('bar')})
    def room_of_requirement():
        if False:
            return 10
        requires_resource_a()
        requires_resource_b()
        not_resources()
    room_of_requirement.execute_in_process()

def test_init_resources():
    if False:
        for i in range(10):
            print('nop')
    resources_initted = {}

    @resource
    def resource_a(_):
        if False:
            i = 10
            return i + 15
        resources_initted['a'] = True
        yield 'A'

    @resource
    def resource_b(_):
        if False:
            print('Hello World!')
        resources_initted['b'] = True
        yield 'B'

    @op
    def consumes_resource_a(a: ResourceParam[str]):
        if False:
            while True:
                i = 10
        assert a == 'A'

    @op
    def consumes_resource_b(b: ResourceParam[str]):
        if False:
            print('Hello World!')
        assert b == 'B'

    @job(resource_defs={'a': resource_a, 'b': resource_b})
    def selective_init_test_job():
        if False:
            i = 10
            return i + 15
        consumes_resource_a()
        consumes_resource_b()
    assert selective_init_test_job.execute_in_process().success
    assert set(resources_initted.keys()) == {'a', 'b'}

def test_ops_with_dependencies():
    if False:
        for i in range(10):
            print('nop')
    completed = set()

    @op
    def first_op(foo: ResourceParam[str]):
        if False:
            i = 10
            return i + 15
        assert foo == 'foo'
        completed.add('first_op')
        return 'hello'

    @op
    def second_op(foo: ResourceParam[str], first_op_result: str):
        if False:
            for i in range(10):
                print('nop')
        assert foo == 'foo'
        assert first_op_result == 'hello'
        completed.add('second_op')
        return first_op_result + ' world'

    @op
    def third_op():
        if False:
            print('Hello World!')
        completed.add('third_op')
        return '!'

    @op
    def fourth_op(context, second_op_result: str, foo: ResourceParam[str], third_op_result: str):
        if False:
            while True:
                i = 10
        assert foo == 'foo'
        assert second_op_result == 'hello world'
        assert third_op_result == '!'
        completed.add('fourth_op')
        return second_op_result + third_op_result

    @job(resource_defs={'foo': ResourceDefinition.hardcoded_resource('foo')})
    def op_dependencies_job():
        if False:
            for i in range(10):
                print('nop')
        fourth_op(second_op_result=second_op(first_op()), third_op_result=third_op())
    assert op_dependencies_job.execute_in_process().success
    assert completed == {'first_op', 'second_op', 'third_op', 'fourth_op'}

def test_assets():
    if False:
        for i in range(10):
            print('nop')
    executed = {}

    @asset
    def the_asset(context, foo: ResourceParam[str]):
        if False:
            while True:
                i = 10
        assert context.resources.foo == 'blah'
        assert foo == 'blah'
        executed['the_asset'] = True
        return 'hello'

    @asset
    def the_other_asset(context, the_asset, foo: ResourceParam[str]):
        if False:
            while True:
                i = 10
        assert context.resources.foo == 'blah'
        assert foo == 'blah'
        assert the_asset == 'hello'
        executed['the_other_asset'] = True
        return 'world'

    @asset
    def the_third_asset(context, the_asset, foo: ResourceParam[str], the_other_asset):
        if False:
            while True:
                i = 10
        assert context.resources.foo == 'blah'
        assert foo == 'blah'
        assert the_asset == 'hello'
        assert the_other_asset == 'world'
        executed['the_third_asset'] = True
    transformed_assets = with_resources([the_asset, the_other_asset, the_third_asset], {'foo': ResourceDefinition.hardcoded_resource('blah')})
    assert build_assets_job('the_job', transformed_assets).execute_in_process().success
    assert executed['the_asset']
    assert executed['the_other_asset']
    assert executed['the_third_asset']

def test_multi_assets():
    if False:
        return 10
    executed = {}

    @multi_asset(outs={'a': AssetOut(key='asset_a'), 'b': AssetOut(key='asset_b')})
    def two_assets(context, foo: ResourceParam[str]):
        if False:
            return 10
        assert context.resources.foo == 'blah'
        assert foo == 'blah'
        executed['two_assets'] = True
        return (1, 2)
    transformed_assets = with_resources([two_assets], {'foo': ResourceDefinition.hardcoded_resource('blah')})[0]
    assert isinstance(transformed_assets, AssetsDefinition)
    assert build_assets_job('the_job', [transformed_assets]).execute_in_process().success
    assert executed['two_assets']

def test_resource_not_provided():
    if False:
        while True:
            i = 10

    @asset
    def consumes_nonexistent_resource(not_provided: ResourceParam[str]):
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'not_provided' required by op 'consumes_nonexistent_resource'"):
        with_resources([consumes_nonexistent_resource], {})

def test_resource_class():
    if False:
        print('Hello World!')
    resource_called = {}

    class MyResource(ResourceDefinition):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__(resource_fn=lambda *_, **__: self)

        def do_something(self):
            if False:
                print('Hello World!')
            resource_called['called'] = True

    @op
    def do_something_op(my_resource: ResourceParam[MyResource]):
        if False:
            i = 10
            return i + 15
        my_resource.do_something()

    @job(resource_defs={'my_resource': MyResource()})
    def my_job():
        if False:
            print('Hello World!')
        do_something_op()
    assert my_job.execute_in_process().success
    assert resource_called['called']

    @asset
    def consumes_nonexistent_resource_class(not_provided: ResourceParam[MyResource]):
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'not_provided' required by op 'consumes_nonexistent_resource_class'"):
        with_resources([consumes_nonexistent_resource_class], {})

def test_both_decorator_and_argument_error():
    if False:
        return 10
    with pytest.raises(ParameterCheckError, match='Invariant violation for parameter Cannot specify resource requirements in both @asset decorator and as arguments to the decorated function'):

        @asset(required_resource_keys={'foo'})
        def my_asset(bar: ResourceParam[Any]):
            if False:
                return 10
            pass
    with pytest.raises(ParameterCheckError, match='Invariant violation for parameter Cannot specify resource requirements in both @multi_asset decorator and as arguments to the decorated function'):

        @multi_asset(outs={'a': AssetOut(key='asset_a'), 'b': AssetOut(key='asset_b')}, required_resource_keys={'foo'})
        def my_assets(bar: ResourceParam[Any]):
            if False:
                print('Hello World!')
            pass
    with pytest.raises(ParameterCheckError, match='Invariant violation for parameter Cannot specify resource requirements in both @op decorator and as arguments to the decorated function'):

        @op(required_resource_keys={'foo'})
        def my_op(bar: ResourceParam[Any]):
            if False:
                i = 10
                return i + 15
            pass

def test_asset_with_structured_config():
    if False:
        print('Hello World!')

    class AnAssetConfig(Config):
        a_string: str
        an_int: int
    executed = {}

    @asset
    def the_asset(context, config: AnAssetConfig, foo: ResourceParam[str]):
        if False:
            i = 10
            return i + 15
        assert context.resources.foo == 'blah'
        assert foo == 'blah'
        assert context.op_config['a_string'] == 'foo'
        assert config.a_string == 'foo'
        assert config.an_int == 2
        executed['the_asset'] = True
    transformed_asset = with_resources([the_asset], {'foo': ResourceDefinition.hardcoded_resource('blah')})[0]
    assert isinstance(transformed_asset, AssetsDefinition)
    assert build_assets_job('the_job', [transformed_asset], config={'ops': {'the_asset': {'config': {'a_string': 'foo', 'an_int': 2}}}}).execute_in_process().success
    assert executed['the_asset']

@pytest.mark.skipif(sys.version_info < (3, 9), reason='requires python3.9')
def test_no_err_builtin_annotations():
    if False:
        i = 10
        return i + 15
    executed = {}

    @asset
    def the_asset(context, foo: ResourceParam[str]):
        if False:
            return 10
        assert context.resources.foo == 'blah'
        assert foo == 'blah'
        executed['the_asset'] = True
        return [{'hello': 'world'}]

    @asset
    def the_other_asset(context, the_asset: list[dict[str, str]], foo: ResourceParam[str]):
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.foo == 'blah'
        assert foo == 'blah'
        assert the_asset == [{'hello': 'world'}]
        executed['the_other_asset'] = True
        return 'world'
    transformed_assets = with_resources([the_asset, the_other_asset], {'foo': ResourceDefinition.hardcoded_resource('blah')})
    assert build_assets_job('the_job', transformed_assets).execute_in_process().success
    assert executed['the_asset']
    assert executed['the_other_asset']