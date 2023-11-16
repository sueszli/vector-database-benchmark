from collections import defaultdict
import pytest
from dagster import DynamicOut, DynamicOutput, Int, Out, Output, graph, job, op, resource
from dagster._core.definitions import failure_hook, success_hook
from dagster._core.definitions.decorators.hook_decorator import event_list_hook
from dagster._core.definitions.events import Failure, HookExecutionResult
from dagster._core.errors import DagsterInvalidDefinitionError

class SomeUserException(Exception):
    pass

@resource
def resource_a(_init_context):
    if False:
        print('Hello World!')
    return 1

def test_hook_on_op_instance():
    if False:
        for i in range(10):
            print('nop')
    called_hook_to_ops = defaultdict(set)

    @event_list_hook(required_resource_keys={'resource_a'})
    def a_hook(context, _):
        if False:
            print('Hello World!')
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        assert context.resources.resource_a == 1
        return HookExecutionResult('a_hook')

    @op
    def a_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @job(resource_defs={'resource_a': resource_a})
    def a_job():
        if False:
            print('Hello World!')
        a_op.with_hooks(hook_defs={a_hook})()
        a_op.alias('op_with_hook').with_hooks(hook_defs={a_hook})()
        a_op.alias('op_without_hook')()
    result = a_job.execute_in_process()
    assert result.success
    assert called_hook_to_ops['a_hook'] == {'a_op', 'op_with_hook'}

def test_hook_accumulation():
    if False:
        i = 10
        return i + 15
    called_hook_to_step_keys = defaultdict(set)

    @event_list_hook
    def job_hook(context, _):
        if False:
            return 10
        called_hook_to_step_keys[context.hook_def.name].add(context.step_key)
        return HookExecutionResult('job_hook')

    @event_list_hook
    def op_1_hook(context, _):
        if False:
            i = 10
            return i + 15
        called_hook_to_step_keys[context.hook_def.name].add(context.step_key)
        return HookExecutionResult('op_1_hook')

    @event_list_hook
    def graph_1_hook(context, _):
        if False:
            for i in range(10):
                print('nop')
        called_hook_to_step_keys[context.hook_def.name].add(context.step_key)
        return HookExecutionResult('graph_1_hook')

    @op
    def op_1(_):
        if False:
            for i in range(10):
                print('nop')
        return 1

    @op
    def op_2(_, num):
        if False:
            for i in range(10):
                print('nop')
        return num

    @op
    def op_3(_):
        if False:
            for i in range(10):
                print('nop')
        return 1

    @graph
    def graph_1():
        if False:
            for i in range(10):
                print('nop')
        return op_2(op_1.with_hooks({op_1_hook})())

    @graph
    def graph_2():
        if False:
            print('Hello World!')
        op_3()
        return graph_1.with_hooks({graph_1_hook})()

    @job_hook
    @job
    def a_job():
        if False:
            return 10
        graph_2()
    result = a_job.execute_in_process()
    assert result.success
    assert called_hook_to_step_keys == {'job_hook': {'graph_2.graph_1.op_1', 'graph_2.graph_1.op_2', 'graph_2.op_3'}, 'op_1_hook': {'graph_2.graph_1.op_1'}, 'graph_1_hook': {'graph_2.graph_1.op_1', 'graph_2.graph_1.op_2'}}

def test_hook_on_graph_instance():
    if False:
        while True:
            i = 10
    called_hook_to_step_keys = defaultdict(set)

    @event_list_hook
    def hook_a_generic(context, _):
        if False:
            print('Hello World!')
        called_hook_to_step_keys[context.hook_def.name].add(context.step_key)
        return HookExecutionResult('hook_a_generic')

    @op
    def two(_):
        if False:
            return 10
        return 1

    @op
    def add_one(_, num):
        if False:
            print('Hello World!')
        return num + 1

    @graph
    def add_two():
        if False:
            while True:
                i = 10
        adder_1 = add_one.alias('adder_1')
        adder_2 = add_one.alias('adder_2')
        return adder_2(adder_1(two()))

    @job
    def a_job():
        if False:
            while True:
                i = 10
        add_two.with_hooks({hook_a_generic})()
    result = a_job.execute_in_process()
    assert result.success
    assert called_hook_to_step_keys['hook_a_generic'] == set([i.step_key for i in result.filter_events(lambda i: i.is_step_event)])

def test_success_hook_on_op_instance():
    if False:
        for i in range(10):
            print('nop')
    called_hook_to_ops = defaultdict(set)

    @success_hook(required_resource_keys={'resource_a'})
    def a_hook(context):
        if False:
            return 10
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        assert context.resources.resource_a == 1

    @op
    def a_op(_):
        if False:
            print('Hello World!')
        pass

    @op
    def failed_op(_):
        if False:
            i = 10
            return i + 15
        raise SomeUserException()

    @job(resource_defs={'resource_a': resource_a})
    def a_job():
        if False:
            while True:
                i = 10
        a_op.with_hooks(hook_defs={a_hook})()
        a_op.alias('op_with_hook').with_hooks(hook_defs={a_hook})()
        a_op.alias('op_without_hook')()
        failed_op.with_hooks(hook_defs={a_hook})()
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert called_hook_to_ops['a_hook'] == {'a_op', 'op_with_hook'}

def test_success_hook_on_op_instance_subset():
    if False:
        for i in range(10):
            print('nop')
    called_hook_to_ops = defaultdict(set)

    @success_hook(required_resource_keys={'resource_a'})
    def a_hook(context):
        if False:
            return 10
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        assert context.resources.resource_a == 1

    @op
    def a_op(_):
        if False:
            print('Hello World!')
        pass

    @op
    def failed_op(_):
        if False:
            for i in range(10):
                print('nop')
        raise SomeUserException()

    @job(resource_defs={'resource_a': resource_a})
    def a_job():
        if False:
            return 10
        a_op.with_hooks(hook_defs={a_hook})()
        a_op.alias('op_with_hook').with_hooks(hook_defs={a_hook})()
        a_op.alias('op_without_hook')()
        failed_op.with_hooks(hook_defs={a_hook})()
    result = a_job.execute_in_process(raise_on_error=False, op_selection=['a_op', 'op_with_hook'])
    assert result.success
    assert called_hook_to_ops['a_hook'] == {'a_op', 'op_with_hook'}

def test_failure_hook_on_op_instance():
    if False:
        for i in range(10):
            print('nop')
    called_hook_to_ops = defaultdict(set)

    @failure_hook(required_resource_keys={'resource_a'})
    def a_hook(context):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        assert context.resources.resource_a == 1

    @op
    def failed_op(_):
        if False:
            return 10
        raise SomeUserException()

    @op
    def a_succeeded_op(_):
        if False:
            while True:
                i = 10
        pass

    @job(resource_defs={'resource_a': resource_a})
    def a_job():
        if False:
            i = 10
            return i + 15
        failed_op.with_hooks(hook_defs={a_hook})()
        failed_op.alias('op_with_hook').with_hooks(hook_defs={a_hook})()
        failed_op.alias('op_without_hook')()
        a_succeeded_op.with_hooks(hook_defs={a_hook})()
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert called_hook_to_ops['a_hook'] == {'failed_op', 'op_with_hook'}

def test_failure_hook_op_exception():
    if False:
        return 10
    called = {}

    @failure_hook
    def a_hook(context):
        if False:
            i = 10
            return i + 15
        called[context.op.name] = context.op_exception

    @op
    def a_op(_):
        if False:
            return 10
        pass

    @op
    def user_code_error_op(_):
        if False:
            while True:
                i = 10
        raise SomeUserException()

    @op
    def failure_op(_):
        if False:
            while True:
                i = 10
        raise Failure()

    @a_hook
    @job
    def a_job():
        if False:
            for i in range(10):
                print('nop')
        a_op()
        user_code_error_op()
        failure_op()
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert 'a_op' not in called
    assert isinstance(called.get('user_code_error_op'), SomeUserException)
    assert isinstance(called.get('failure_op'), Failure)

def test_none_op_exception_access():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    @success_hook
    def a_hook(context):
        if False:
            i = 10
            return i + 15
        called[context.op.name] = context.op_exception

    @op
    def a_op(_):
        if False:
            print('Hello World!')
        pass

    @a_hook
    @job
    def a_job():
        if False:
            i = 10
            return i + 15
        a_op()
    result = a_job.execute_in_process(raise_on_error=False)
    assert result.success
    assert called.get('a_op') is None

def test_op_outputs_access():
    if False:
        i = 10
        return i + 15
    called = {}

    @success_hook
    def my_success_hook(context):
        if False:
            for i in range(10):
                print('nop')
        called[context.step_key] = context.op_output_values

    @failure_hook
    def my_failure_hook(context):
        if False:
            for i in range(10):
                print('nop')
        called[context.step_key] = context.op_output_values

    @op(out={'one': Out(), 'two': Out(), 'three': Out()})
    def a_op(_):
        if False:
            print('Hello World!')
        yield Output(1, 'one')
        yield Output(2, 'two')
        yield Output(3, 'three')

    @op(out={'one': Out(), 'two': Out()})
    def failed_op(_):
        if False:
            while True:
                i = 10
        yield Output(1, 'one')
        raise SomeUserException()
        yield Output(3, 'two')

    @op(out=DynamicOut())
    def dynamic_op(_):
        if False:
            i = 10
            return i + 15
        yield DynamicOutput(1, mapping_key='mapping_1')
        yield DynamicOutput(2, mapping_key='mapping_2')

    @op
    def echo(_, x):
        if False:
            for i in range(10):
                print('nop')
        return x

    @my_success_hook
    @my_failure_hook
    @job
    def a_job():
        if False:
            return 10
        a_op()
        failed_op()
        dynamic_op().map(echo)
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert called.get('a_op') == {'one': 1, 'two': 2, 'three': 3}
    assert called.get('failed_op') == {'one': 1}
    assert called.get('dynamic_op') == {'result': {'mapping_1': 1, 'mapping_2': 2}}
    assert called.get('echo[mapping_1]') == {'result': 1}
    assert called.get('echo[mapping_2]') == {'result': 2}

def test_hook_on_job_def():
    if False:
        return 10
    called_hook_to_ops = defaultdict(set)

    @event_list_hook
    def hook_a_generic(context, _):
        if False:
            while True:
                i = 10
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('hook_a_generic')

    @event_list_hook
    def hook_b_generic(context, _):
        if False:
            return 10
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('hook_b_generic')

    @op
    def op_a(_):
        if False:
            print('Hello World!')
        pass

    @op
    def op_b(_):
        if False:
            print('Hello World!')
        pass

    @op
    def op_c(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @job(hooks={hook_b_generic})
    def a_job():
        if False:
            print('Hello World!')
        op_a()
        op_b()
        op_c()
    result = a_job.with_hooks({hook_a_generic}).execute_in_process()
    assert result.success
    assert called_hook_to_ops == {'hook_b_generic': {'op_b', 'op_a', 'op_c'}, 'hook_a_generic': {'op_b', 'op_a', 'op_c'}}

def test_hook_on_job_def_with_graphs():
    if False:
        for i in range(10):
            print('nop')
    called_hook_to_step_keys = defaultdict(set)

    @event_list_hook
    def hook_a_generic(context, _):
        if False:
            print('Hello World!')
        called_hook_to_step_keys[context.hook_def.name].add(context.step_key)
        return HookExecutionResult('hook_a_generic')

    @op
    def two(_):
        if False:
            i = 10
            return i + 15
        return 1

    @op
    def add_one(_, num):
        if False:
            print('Hello World!')
        return num + 1

    @graph
    def add_two():
        if False:
            for i in range(10):
                print('nop')
        adder_1 = add_one.alias('adder_1')
        adder_2 = add_one.alias('adder_2')
        return adder_2(adder_1(two()))

    @job
    def a_job():
        if False:
            for i in range(10):
                print('nop')
        add_two()
    hooked_job = a_job.with_hooks({hook_a_generic})
    assert hooked_job.all_node_defs == a_job.all_node_defs
    result = hooked_job.execute_in_process()
    assert result.success
    assert called_hook_to_step_keys['hook_a_generic'] == set([i.step_key for i in result.filter_events(lambda i: i.is_step_event)])

def test_hook_decorate_job_def():
    if False:
        return 10
    called_hook_to_ops = defaultdict(set)

    @event_list_hook
    def hook_a_generic(context, _):
        if False:
            for i in range(10):
                print('nop')
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('hook_a_generic')

    @success_hook
    def hook_b_success(context):
        if False:
            print('Hello World!')
        called_hook_to_ops[context.hook_def.name].add(context.op.name)

    @failure_hook
    def hook_c_failure(context):
        if False:
            while True:
                i = 10
        called_hook_to_ops[context.hook_def.name].add(context.op.name)

    @op
    def op_a(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @op
    def op_b(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @op
    def failed_op(_):
        if False:
            print('Hello World!')
        raise SomeUserException()

    @hook_c_failure
    @hook_b_success
    @hook_a_generic
    @job
    def a_job():
        if False:
            for i in range(10):
                print('nop')
        op_a()
        failed_op()
        op_b()
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert called_hook_to_ops['hook_a_generic'] == {'op_a', 'op_b', 'failed_op'}
    assert called_hook_to_ops['hook_b_success'] == {'op_a', 'op_b'}
    assert called_hook_to_ops['hook_c_failure'] == {'failed_op'}

def test_hook_on_job_def_and_op_instance():
    if False:
        return 10
    called_hook_to_ops = defaultdict(set)

    @event_list_hook
    def hook_a_generic(context, _):
        if False:
            for i in range(10):
                print('nop')
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('hook_a_generic')

    @success_hook
    def hook_b_success(context):
        if False:
            print('Hello World!')
        called_hook_to_ops[context.hook_def.name].add(context.op.name)

    @failure_hook
    def hook_c_failure(context):
        if False:
            for i in range(10):
                print('nop')
        called_hook_to_ops[context.hook_def.name].add(context.op.name)

    @op
    def op_a(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @op
    def op_b(_):
        if False:
            while True:
                i = 10
        pass

    @op
    def failed_op(_):
        if False:
            print('Hello World!')
        raise SomeUserException()

    @hook_a_generic
    @job
    def a_job():
        if False:
            i = 10
            return i + 15
        op_a.with_hooks({hook_b_success})()
        failed_op.with_hooks({hook_c_failure})()
        op_b.with_hooks({hook_a_generic})()
    result = a_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert called_hook_to_ops['hook_a_generic'] == {'op_a', 'op_b', 'failed_op'}
    assert called_hook_to_ops['hook_b_success'] == {'op_a'}
    assert called_hook_to_ops['hook_c_failure'] == {'failed_op'}
    hook_events = result.filter_events(lambda event: event.is_hook_event)
    assert len(hook_events) == 5

def test_hook_context_config_schema():
    if False:
        for i in range(10):
            print('nop')
    called_hook_to_ops = defaultdict(set)

    @event_list_hook
    def a_hook(context, _):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        assert context.op_config == {'config_1': 1}
        return HookExecutionResult('a_hook')

    @op(config_schema={'config_1': Int})
    def a_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @job
    def a_job():
        if False:
            return 10
        a_op.with_hooks(hook_defs={a_hook})()
    result = a_job.execute_in_process(run_config={'ops': {'a_op': {'config': {'config_1': 1}}}})
    assert result.success
    assert called_hook_to_ops['a_hook'] == {'a_op'}

def test_hook_resource_mismatch():
    if False:
        print('Hello World!')

    @event_list_hook(required_resource_keys={'b'})
    def a_hook(context, _):
        if False:
            while True:
                i = 10
        assert context.resources.resource_a == 1
        return HookExecutionResult('a_hook')

    @op
    def a_op(_):
        if False:
            return 10
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'b' required by hook 'a_hook' attached to job '_' was not provided"):

        @a_hook
        @job(resource_defs={'a': resource_a})
        def _():
            if False:
                return 10
            a_op()
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'b' required by hook 'a_hook' attached to op 'a_op' was not provided"):

        @job(resource_defs={'a': resource_a})
        def _():
            if False:
                for i in range(10):
                    print('nop')
            a_op.with_hooks({a_hook})()

def test_hook_subjob():
    if False:
        print('Hello World!')
    called_hook_to_ops = defaultdict(set)

    @event_list_hook
    def hook_a_generic(context, _):
        if False:
            for i in range(10):
                print('nop')
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('hook_a_generic')

    @op
    def op_a(_):
        if False:
            print('Hello World!')
        pass

    @op
    def op_b(_):
        if False:
            i = 10
            return i + 15
        pass

    @hook_a_generic
    @job
    def a_job():
        if False:
            return 10
        op_a()
        op_b()
    result = a_job.execute_in_process()
    assert result.success
    assert called_hook_to_ops['hook_a_generic'] == {'op_a', 'op_b'}
    called_hook_to_ops = defaultdict(set)
    result = a_job.execute_in_process(op_selection=['op_a'])
    assert result.success
    assert called_hook_to_ops['hook_a_generic'] == {'op_a'}

def test_hook_ops():
    if False:
        while True:
            i = 10
    called_hook_to_ops = defaultdict(set)

    @success_hook
    def my_hook(context):
        if False:
            return 10
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('my_hook')

    @op
    def a_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @graph
    def a_graph():
        if False:
            print('Hello World!')
        a_op.with_hooks(hook_defs={my_hook})()
        a_op.alias('op_with_hook').with_hooks(hook_defs={my_hook})()
        a_op.alias('op_without_hook')()
    result = a_graph.execute_in_process()
    assert result.success
    assert called_hook_to_ops['my_hook'] == {'a_op', 'op_with_hook'}

def test_hook_graph():
    if False:
        return 10
    called_hook_to_ops = defaultdict(set)

    @success_hook
    def a_hook(context):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('a_hook')

    @success_hook
    def b_hook(context):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('a_hook')

    @op
    def a_op(_):
        if False:
            print('Hello World!')
        pass

    @op
    def b_op(_):
        if False:
            i = 10
            return i + 15
        pass

    @a_hook
    @graph
    def sub_graph():
        if False:
            for i in range(10):
                print('nop')
        a_op()

    @b_hook
    @graph
    def super_graph():
        if False:
            print('Hello World!')
        sub_graph()
        b_op()
    result = super_graph.execute_in_process()
    assert result.success
    assert called_hook_to_ops['a_hook'] == {'a_op'}
    assert called_hook_to_ops['b_hook'] == {'a_op', 'b_op'}
    called_hook_to_ops = defaultdict(set)
    result = super_graph.to_job().execute_in_process()
    assert result.success
    assert called_hook_to_ops['a_hook'] == {'a_op'}
    assert called_hook_to_ops['b_hook'] == {'a_op', 'b_op'}

def test_hook_on_job():
    if False:
        print('Hello World!')
    called_hook_to_ops = defaultdict(set)

    @success_hook
    def a_hook(context):
        if False:
            i = 10
            return i + 15
        called_hook_to_ops[context.hook_def.name].add(context.op.name)
        return HookExecutionResult('a_hook')

    @op
    def basic():
        if False:
            i = 10
            return i + 15
        return 5

    @a_hook
    @job
    def hooked_job():
        if False:
            for i in range(10):
                print('nop')
        basic()
        basic()
        basic()
    result = hooked_job.execute_in_process()
    assert result.success
    assert called_hook_to_ops['a_hook'] == {'basic', 'basic_2', 'basic_3'}