import pytest
from dagster import InitResourceContext, build_init_resource_context, resource
from dagster._core.errors import DagsterInvariantViolationError

def test_build_no_args():
    if False:
        return 10
    context = build_init_resource_context()
    assert isinstance(context, InitResourceContext)

    @resource
    def basic(_):
        if False:
            print('Hello World!')
        return 'foo'
    assert basic(context) == 'foo'

def test_build_with_resources():
    if False:
        for i in range(10):
            print('nop')

    @resource
    def foo(_):
        if False:
            return 10
        return 'foo'
    context = build_init_resource_context(resources={'foo': foo, 'bar': 'bar'})
    assert context.resources.foo == 'foo'
    assert context.resources.bar == 'bar'

    @resource(required_resource_keys={'foo', 'bar'})
    def reqs_resources(context):
        if False:
            print('Hello World!')
        return context.resources.foo + context.resources.bar
    assert reqs_resources(context) == 'foobar'

def test_build_with_cm_resource():
    if False:
        i = 10
        return i + 15
    entered = []

    @resource
    def foo(_):
        if False:
            return 10
        try:
            yield 'foo'
        finally:
            entered.append('true')

    @resource(required_resource_keys={'foo'})
    def reqs_cm_resource(context):
        if False:
            i = 10
            return i + 15
        return context.resources.foo + 'bar'
    context = build_init_resource_context(resources={'foo': foo})
    with pytest.raises(DagsterInvariantViolationError):
        context.resources
    del context
    assert entered == ['true']
    with build_init_resource_context(resources={'foo': foo}) as context:
        assert context.resources.foo == 'foo'
        assert reqs_cm_resource(context) == 'foobar'
    assert entered == ['true', 'true']