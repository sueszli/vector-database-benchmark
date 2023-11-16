from collections import defaultdict
from unittest import mock
import pytest
from dagster import DagsterEventType, GraphDefinition, JobDefinition, NodeInvocation, build_hook_context, execute_job, graph, job, op, reconstructable, resource
from dagster._core.definitions import NodeHandle, failure_hook, success_hook
from dagster._core.definitions.decorators.hook_decorator import event_list_hook
from dagster._core.definitions.events import HookExecutionResult
from dagster._core.definitions.policy import RetryPolicy
from dagster._core.errors import DagsterExecutionInterruptedError, DagsterInvalidDefinitionError
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import instance_for_test

class SomeUserException(Exception):
    pass

@resource
def resource_a(_init_context):
    if False:
        print('Hello World!')
    return 1

def test_hook():
    if False:
        while True:
            i = 10
    called = {}

    @event_list_hook
    def a_hook(context, event_list):
        if False:
            while True:
                i = 10
        called[context.hook_def.name] = context.op.name
        called['step_event_list'] = [i for i in event_list]
        return HookExecutionResult(hook_name='a_hook')

    @event_list_hook(name='a_named_hook')
    def named_hook(context, _):
        if False:
            return 10
        called[context.hook_def.name] = context.op.name
        return HookExecutionResult(hook_name='a_hook')

    @op
    def a_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass
    a_job = GraphDefinition(node_defs=[a_op], name='test', dependencies={NodeInvocation('a_op', 'a_op_with_hook', hook_defs={a_hook, named_hook}): {}})
    result = a_job.execute_in_process()
    assert result.success
    assert called.get('a_hook') == 'a_op_with_hook'
    assert called.get('a_named_hook') == 'a_op_with_hook'
    assert set([event.event_type_value for event in called['step_event_list']]) == set([event.event_type_value for event in result.filter_events(lambda event: event.is_step_event)])

def test_hook_user_error():
    if False:
        i = 10
        return i + 15

    @event_list_hook
    def error_hook(context, _):
        if False:
            i = 10
            return i + 15
        raise SomeUserException()

    @op
    def a_op(_):
        if False:
            return 10
        return 1
    a_job = GraphDefinition(node_defs=[a_op], name='test', dependencies={NodeInvocation('a_op', 'a_op_with_hook', hook_defs={error_hook}): {}})
    result = a_job.execute_in_process()
    assert result.success
    hook_errored_events = result.filter_events(lambda event: event.event_type == DagsterEventType.HOOK_ERRORED)
    assert len(hook_errored_events) == 1
    assert hook_errored_events[0].node_handle.name == 'a_op_with_hook'

def test_hook_decorator_arg_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(DagsterInvalidDefinitionError, match='does not have required positional'):

        @success_hook
        def _():
            if False:
                return 10
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match='does not have required positional'):

        @failure_hook
        def _():
            if False:
                print('Hello World!')
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match='does not have required positional'):

        @event_list_hook()
        def _(_):
            if False:
                i = 10
                return i + 15
            pass

def test_hook_with_resource():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    @event_list_hook(required_resource_keys={'resource_a'})
    def a_hook(context, _):
        if False:
            print('Hello World!')
        called[context.op.name] = True
        assert context.resources.resource_a == 1
        return HookExecutionResult(hook_name='a_hook')

    @op
    def a_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass
    a_job = GraphDefinition(node_defs=[a_op], name='test', dependencies={NodeInvocation('a_op', 'a_op_with_hook', hook_defs={a_hook}): {}}).to_job(resource_defs={'resource_a': resource_a})
    result = a_job.execute_in_process()
    assert result.success
    assert called.get('a_op_with_hook')

def test_hook_resource_error():
    if False:
        return 10

    @event_list_hook(required_resource_keys={'resource_b'})
    def a_hook(context, event_list):
        if False:
            while True:
                i = 10
        return HookExecutionResult(hook_name='a_hook')

    @op
    def a_op(_):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'resource_b' required by hook 'a_hook' attached to op 'a_op_with_hook' was not provided"):
        GraphDefinition(node_defs=[a_op], name='test', dependencies={NodeInvocation('a_op', 'a_op_with_hook', hook_defs={a_hook}): {}}).to_job(resource_defs={'resource_a': resource_a})

def test_success_hook():
    if False:
        print('Hello World!')
    called_hook_to_ops = defaultdict(list)

    @success_hook
    def a_success_hook(context):
        if False:
            return 10
        called_hook_to_ops[context.hook_def.name].append(context.op.name)

    @success_hook(name='a_named_success_hook')
    def named_success_hook(context):
        if False:
            return 10
        called_hook_to_ops[context.hook_def.name].append(context.op.name)

    @success_hook(required_resource_keys={'resource_a'})
    def success_hook_resource(context):
        if False:
            while True:
                i = 10
        called_hook_to_ops[context.hook_def.name].append(context.op.name)
        assert context.resources.resource_a == 1

    @op
    def succeeded_op(_):
        if False:
            return 10
        pass

    @op
    def failed_op(_):
        if False:
            i = 10
            return i + 15
        raise SomeUserException()
    a_job = GraphDefinition(node_defs=[succeeded_op, failed_op], name='test', dependencies={NodeInvocation('succeeded_op', 'succeeded_op_with_hook', hook_defs={a_success_hook, named_success_hook, success_hook_resource}): {}, NodeInvocation('failed_op', 'failed_op_with_hook', hook_defs={a_success_hook, named_success_hook}): {}}).to_job(resource_defs={'resource_a': resource_a})
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert 'succeeded_op_with_hook' in called_hook_to_ops['a_success_hook']
    assert 'succeeded_op_with_hook' in called_hook_to_ops['a_named_success_hook']
    assert 'succeeded_op_with_hook' in called_hook_to_ops['success_hook_resource']
    assert 'failed_op_with_hook' not in called_hook_to_ops['a_success_hook']
    assert 'failed_op_with_hook' not in called_hook_to_ops['a_named_success_hook']

def test_failure_hook():
    if False:
        i = 10
        return i + 15
    called_hook_to_ops = defaultdict(list)

    @failure_hook
    def a_failure_hook(context):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(context.instance, DagsterInstance)
        called_hook_to_ops[context.hook_def.name].append(context.op.name)

    @failure_hook(name='a_named_failure_hook')
    def named_failure_hook(context):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].append(context.op.name)

    @failure_hook(required_resource_keys={'resource_a'})
    def failure_hook_resource(context):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].append(context.op.name)
        assert context.resources.resource_a == 1

    @op
    def succeeded_op(_):
        if False:
            while True:
                i = 10
        pass

    @op
    def failed_op(_):
        if False:
            print('Hello World!')
        raise SomeUserException()
    a_job = GraphDefinition(node_defs=[failed_op, succeeded_op], name='test', dependencies={NodeInvocation('failed_op', 'failed_op_with_hook', hook_defs={a_failure_hook, named_failure_hook, failure_hook_resource}): {}, NodeInvocation('succeeded_op', 'succeeded_op_with_hook', hook_defs={a_failure_hook, named_failure_hook}): {}}).to_job(resource_defs={'resource_a': resource_a})
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert 'failed_op_with_hook' in called_hook_to_ops['a_failure_hook']
    assert 'failed_op_with_hook' in called_hook_to_ops['a_named_failure_hook']
    assert 'failed_op_with_hook' in called_hook_to_ops['failure_hook_resource']
    assert 'succeeded_op_with_hook' not in called_hook_to_ops['a_failure_hook']
    assert 'succeeded_op_with_hook' not in called_hook_to_ops['a_named_failure_hook']

def test_failure_hook_framework_exception():
    if False:
        print('Hello World!')
    called_hook_to_ops = defaultdict(list)

    @failure_hook
    def a_failure_hook(context):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].append(context.op.name)

    @op
    def my_op(_):
        if False:
            print('Hello World!')
        pass

    @job(hooks={a_failure_hook})
    def my_job():
        if False:
            while True:
                i = 10
        my_op()
    with mock.patch('dagster._core.execution.plan.execute_plan.core_dagster_event_sequence_for_step') as mocked_event_sequence:
        mocked_event_sequence.side_effect = Exception('Framework exception during execution')
        result = my_job.execute_in_process(raise_on_error=False)
        assert not result.success
        assert 'my_op' in called_hook_to_ops['a_failure_hook']
        called_hook_to_ops = defaultdict(list)
        mocked_event_sequence.side_effect = DagsterExecutionInterruptedError('Execution interrupted during execution')
        result = my_job.execute_in_process(raise_on_error=False)
        assert not result.success
        assert 'my_op' not in called_hook_to_ops['a_failure_hook']

def test_success_hook_event():
    if False:
        for i in range(10):
            print('nop')

    @success_hook
    def a_hook(_):
        if False:
            while True:
                i = 10
        pass

    @op
    def a_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @op
    def failed_op(_):
        if False:
            print('Hello World!')
        raise SomeUserException()
    a_job = GraphDefinition(node_defs=[a_op, failed_op], name='test', dependencies={NodeInvocation('a_op', hook_defs={a_hook}): {}, NodeInvocation('failed_op', hook_defs={a_hook}): {}})
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    hook_events = result.filter_events(lambda event: event.is_hook_event)
    assert len(hook_events) == 2
    for event in hook_events:
        if event.event_type == DagsterEventType.HOOK_COMPLETED:
            assert event.node_name == 'a_op'
        if event.event_type == DagsterEventType.HOOK_SKIPPED:
            assert event.node_name == 'failed_op'

def test_failure_hook_event():
    if False:
        return 10

    @failure_hook
    def a_hook(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @op
    def a_op(_):
        if False:
            while True:
                i = 10
        pass

    @op
    def failed_op(_):
        if False:
            i = 10
            return i + 15
        raise SomeUserException()
    a_job = GraphDefinition(node_defs=[a_op, failed_op], name='test', dependencies={NodeInvocation('a_op', hook_defs={a_hook}): {}, NodeInvocation('failed_op', hook_defs={a_hook}): {}})
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    hook_events = result.filter_events(lambda event: event.is_hook_event)
    assert len(hook_events) == 2
    for event in hook_events:
        if event.event_type == DagsterEventType.HOOK_COMPLETED:
            assert event.node_name == 'failed_op'
        if event.event_type == DagsterEventType.HOOK_SKIPPED:
            assert event.node_name == 'a_op'

@op
def noop(_):
    if False:
        i = 10
        return i + 15
    return

@success_hook
def noop_hook(_):
    if False:
        print('Hello World!')
    return

@noop_hook
@job
def foo():
    if False:
        return 10
    noop()

def test_jobs_with_hooks_are_reconstructable():
    if False:
        while True:
            i = 10
    assert reconstructable(foo)

def test_hook_decorator():
    if False:
        print('Hello World!')
    called_hook_to_ops = defaultdict(list)

    @success_hook
    def a_success_hook(context):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].append(context.op.name)

    @op
    def a_op(_):
        if False:
            return 10
        pass

    @a_success_hook
    @job(description='i am a job', op_retry_policy=RetryPolicy(max_retries=3), tags={'foo': 'FOO'})
    def a_job():
        if False:
            i = 10
            return i + 15
        a_op()
    assert isinstance(a_job, JobDefinition)
    assert a_job.tags
    assert a_job.tags.get('foo') == 'FOO'
    assert a_job.tags.get('foo') == 'FOO'
    assert a_job.description == 'i am a job'
    retry_policy = a_job.get_retry_policy_for_handle(NodeHandle('a_op', parent=None))
    assert isinstance(retry_policy, RetryPolicy)
    assert retry_policy.max_retries == 3

def test_hook_with_resource_to_resource_dep():
    if False:
        i = 10
        return i + 15
    called = {}

    @resource(required_resource_keys={'resource_a'})
    def resource_b(context):
        if False:
            for i in range(10):
                print('nop')
        return context.resources.resource_a

    @event_list_hook(required_resource_keys={'resource_b'})
    def hook_requires_b(context, _):
        if False:
            while True:
                i = 10
        called[context.op.name] = True
        assert context.resources.resource_b == 1
        return HookExecutionResult(hook_name='a_hook')

    @op
    def basic_op():
        if False:
            return 10
        pass

    @job(resource_defs={'resource_a': resource_a, 'resource_b': resource_b})
    def basic_job():
        if False:
            return 10
        basic_op.with_hooks({hook_requires_b})()
    result = basic_job.execute_in_process()
    assert result.success
    assert called.get('basic_op')

    @job(resource_defs={'resource_a': resource_a, 'resource_b': resource_b})
    def basic_job_gonna_use_hooks():
        if False:
            while True:
                i = 10
        basic_op()
    called = {}
    basic_hook_job = basic_job_gonna_use_hooks.with_hooks({hook_requires_b})
    result = basic_hook_job.execute_in_process()
    assert result.success
    assert called.get('basic_op')

def test_hook_graph_job_op():
    if False:
        while True:
            i = 10
    called = {}
    op_output = 'hook_op_output'

    @success_hook(required_resource_keys={'resource_a'})
    def hook_one(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.op.name
        called[context.hook_def.name] = called.get(context.hook_def.name, 0) + 1

    @success_hook()
    def hook_two(context):
        if False:
            while True:
                i = 10
        assert not context.op_config
        assert not context.op_exception
        assert context.op_output_values['result'] == op_output
        called[context.hook_def.name] = called.get(context.hook_def.name, 0) + 1

    @op
    def hook_op(_):
        if False:
            i = 10
            return i + 15
        return op_output
    ctx = build_hook_context(resources={'resource_a': resource_a}, op=hook_op)
    hook_one(ctx)
    assert called.get('hook_one') == 1

    @graph
    def run_success_hook():
        if False:
            return 10
        hook_op.with_hooks({hook_one, hook_two})()
    success_hook_job = run_success_hook.to_job(resource_defs={'resource_a': resource_a})
    assert success_hook_job.execute_in_process().success
    assert called.get('hook_one') == 2
    assert called.get('hook_two') == 1

@success_hook(required_resource_keys={'resource_a'})
def res_hook(context):
    if False:
        while True:
            i = 10
    assert context.resources.resource_a == 1

@op
def emit():
    if False:
        print('Hello World!')
    return 1

@graph
def nested():
    if False:
        return 10
    emit.with_hooks({res_hook})()

@graph
def nested_two():
    if False:
        return 10
    nested()

@job(resource_defs={'resource_a': resource_a})
def res_hook_job():
    if False:
        for i in range(10):
            print('nop')
    nested_two()

def test_multiproc_hook_resource_deps():
    if False:
        return 10
    assert nested.execute_in_process(resources={'resource_a': resource_a}).success
    assert res_hook_job.execute_in_process().success
    with instance_for_test() as instance:
        assert execute_job(reconstructable(res_hook_job), instance=instance).success

def test_hook_decorator_graph_job_op():
    if False:
        i = 10
        return i + 15
    called_hook_to_ops = defaultdict(list)

    @success_hook
    def a_success_hook(context):
        if False:
            print('Hello World!')
        called_hook_to_ops[context.hook_def.name].append(context.op.name)

    @op
    def my_op(_):
        if False:
            while True:
                i = 10
        pass

    @graph
    def a_graph():
        if False:
            i = 10
            return i + 15
        my_op()
    assert a_graph.to_job(hooks={a_success_hook}).execute_in_process().success
    assert called_hook_to_ops['a_success_hook'][0] == 'my_op'

def test_job_hook_context_job_name():
    if False:
        return 10
    my_job_name = 'my_test_job_name'

    @success_hook
    def a_success_hook(context):
        if False:
            i = 10
            return i + 15
        assert context.job_name == my_job_name

    @graph
    def a_graph():
        if False:
            print('Hello World!')
        pass
    assert a_graph.to_job(name=my_job_name, hooks={a_success_hook}).execute_in_process().success