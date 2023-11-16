import re
import pytest
from dagster import AssetMaterialization, InputContext, OutputContext, build_input_context, build_output_context, resource
from dagster._core.errors import DagsterInvariantViolationError

def test_basic_build_input_context():
    if False:
        i = 10
        return i + 15
    context = build_input_context()
    assert isinstance(context, InputContext)

def test_build_input_context_with_resources():
    if False:
        print('Hello World!')

    @resource
    def foo_def():
        if False:
            for i in range(10):
                print('nop')
        return 'bar_def'
    context = build_input_context(resources={'foo': 'bar', 'foo_def': foo_def})
    assert context.resources.foo == 'bar'
    assert context.resources.foo_def == 'bar_def'

def test_build_input_context_with_cm_resource():
    if False:
        i = 10
        return i + 15
    entered = []

    @resource
    def cm_resource():
        if False:
            for i in range(10):
                print('nop')
        try:
            yield 'foo'
        finally:
            entered.append('yes')
    context = build_input_context(resources={'cm_resource': cm_resource})
    with pytest.raises(DagsterInvariantViolationError, match=re.escape('At least one provided resource is a generator, but attempting to access resources outside of context manager scope. You can use the following syntax to open a context manager: `with build_input_context(...) as context:`')):
        context.resources
    del context
    assert entered == ['yes']
    with build_input_context(resources={'cm_resource': cm_resource}) as context:
        assert context.resources.cm_resource == 'foo'
    assert entered == ['yes', 'yes']

def test_basic_build_output_context():
    if False:
        print('Hello World!')
    context = build_output_context()
    assert isinstance(context, OutputContext)

def test_build_output_context_with_cm_resource():
    if False:
        while True:
            i = 10
    entered = []

    @resource
    def cm_resource():
        if False:
            for i in range(10):
                print('nop')
        try:
            yield 'foo'
        finally:
            entered.append('yes')
    context = build_output_context(step_key='test', name='test', resources={'cm_resource': cm_resource})
    with pytest.raises(DagsterInvariantViolationError, match=re.escape('At least one provided resource is a generator, but attempting to access resources outside of context manager scope. You can use the following syntax to open a context manager: `with build_output_context(...) as context:`')):
        context.resources
    del context
    assert entered == ['yes']
    with build_output_context(step_key='test', name='test', resources={'cm_resource': cm_resource}) as context:
        assert context.resources.cm_resource == 'foo'
    assert entered == ['yes', 'yes']

def test_context_logging_user_events():
    if False:
        return 10
    context = build_output_context()
    context.log_event(AssetMaterialization('first'))
    context.log_event(AssetMaterialization('second'))
    assert [event.label for event in context.get_logged_events()] == ['first', 'second']

def test_context_logging_metadata():
    if False:
        while True:
            i = 10
    context = build_output_context()
    context.add_output_metadata({'foo': 'bar'})
    assert 'foo' in context.get_logged_metadata()

def test_output_context_partition_key():
    if False:
        return 10
    context = build_output_context(partition_key='foo')
    assert context.partition_key == 'foo'
    assert context.has_partition_key

def test_input_context_partition_key():
    if False:
        for i in range(10):
            print('nop')
    context = build_input_context(partition_key='foo')
    assert context.partition_key == 'foo'
    assert context.has_partition_key
    context = build_input_context()
    assert not context.has_partition_key