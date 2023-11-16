import asyncio
from datetime import datetime
from functools import partial
import pendulum
import pytest
from dagster import AssetKey, AssetMaterialization, AssetObservation, DailyPartitionsDefinition, DynamicOut, DynamicOutput, ExpectationResult, Failure, Field, In, MultiPartitionsDefinition, Noneable, Nothing, Out, Output, RetryRequested, Selector, StaticPartitionsDefinition, TimeWindow, asset, build_op_context, graph, job, op, resource
from dagster._core.definitions.partition_key_range import PartitionKeyRange
from dagster._core.definitions.time_window_partitions import get_time_partitions_def
from dagster._core.errors import DagsterInvalidConfigError, DagsterInvalidDefinitionError, DagsterInvalidInvocationError, DagsterInvalidPropertyError, DagsterInvariantViolationError, DagsterResourceFunctionError, DagsterStepOutputNotFoundError, DagsterTypeCheckDidNotPass
from dagster._core.execution.context.compute import AssetExecutionContext, OpExecutionContext
from dagster._core.execution.context.invocation import build_asset_context
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_op_invocation_no_arg():
    if False:
        for i in range(10):
            print('nop')

    @op
    def basic_op():
        if False:
            i = 10
            return i + 15
        return 5
    result = basic_op()
    assert result == 5
    basic_op(build_op_context())
    with pytest.raises(DagsterInvalidInvocationError, match="Too many input arguments were provided for op 'basic_op'. This may be because an argument was provided for the context parameter, but no context parameter was defined for the op."):
        basic_op(None)
    with pytest.raises(DagsterInvalidInvocationError, match="Too many input arguments were provided for op 'aliased_basic_op'. This may be because an argument was provided for the context parameter, but no context parameter was defined for the op."):
        basic_op.alias('aliased_basic_op')(None)

def test_op_invocation_none_arg():
    if False:
        i = 10
        return i + 15

    @op
    def basic_op(_):
        if False:
            for i in range(10):
                print('nop')
        return 5
    result = basic_op(None)
    assert result == 5

def test_op_invocation_lifecycle():
    if False:
        while True:
            i = 10

    @op
    def basic_op(context):
        if False:
            print('Hello World!')
        return 5
    with build_op_context() as context:
        pass
    assert context.instance.run_storage._held_conn.closed

def test_op_invocation_context_arg():
    if False:
        for i in range(10):
            print('nop')

    @op
    def basic_op(context):
        if False:
            while True:
                i = 10
        context.log.info('yay')
    basic_op(None)
    basic_op(build_op_context())
    basic_op(context=None)
    basic_op(context=build_op_context())

def test_op_invocation_empty_run_config():
    if False:
        return 10

    @op
    def basic_op(context):
        if False:
            i = 10
            return i + 15
        assert context.run_config is not None
        assert context.run_config == {'resources': {}}
    basic_op(context=build_op_context())

def test_op_invocation_run_config_with_config():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema={'foo': str})
    def basic_op(context):
        if False:
            while True:
                i = 10
        assert context.run_config
        assert context.run_config['ops'] == {'basic_op': {'config': {'foo': 'bar'}}}
    basic_op(build_op_context(op_config={'foo': 'bar'}))

def test_op_invocation_out_of_order_input_defs():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={'x': In(), 'y': In()})
    def check_correct_order(y, x):
        if False:
            while True:
                i = 10
        assert y == 6
        assert x == 5
    check_correct_order(6, 5)
    check_correct_order(x=5, y=6)
    check_correct_order(6, x=5)

def test_op_invocation_with_resources():
    if False:
        for i in range(10):
            print('nop')

    @op(required_resource_keys={'foo'})
    def op_requires_resources(context):
        if False:
            print('Hello World!')
        assert context.resources.foo == 'bar'
        return context.resources.foo
    with pytest.raises(DagsterInvalidInvocationError, match="Decorated function 'op_requires_resources' has context argument, but no context was provided when invoking."):
        op_requires_resources()
    with pytest.raises(DagsterInvalidInvocationError, match="Decorated function 'op_requires_resources' has context argument, but no context was provided when invoking."):
        op_requires_resources.alias('aliased_op_requires_resources')()
    context = build_op_context()
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by op 'op_requires_resources' was not provided"):
        op_requires_resources(context)
    context = build_op_context(resources={'foo': 'bar'})
    assert op_requires_resources(context) == 'bar'

def test_op_invocation_with_cm_resource():
    if False:
        return 10
    teardown_log = []

    @resource
    def cm_resource(_):
        if False:
            while True:
                i = 10
        try:
            yield 'foo'
        finally:
            teardown_log.append('collected')

    @op(required_resource_keys={'cm_resource'})
    def op_requires_cm_resource(context):
        if False:
            return 10
        return context.resources.cm_resource
    context = build_op_context(resources={'cm_resource': cm_resource})
    with pytest.raises(DagsterInvariantViolationError):
        op_requires_cm_resource(context)
    del context
    assert teardown_log == ['collected']
    with build_op_context(resources={'cm_resource': cm_resource}) as context:
        assert op_requires_cm_resource(context) == 'foo'
    assert teardown_log == ['collected', 'collected']

def test_op_invocation_with_config():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema={'foo': str})
    def op_requires_config(context):
        if False:
            while True:
                i = 10
        assert context.op_config['foo'] == 'bar'
        return 5
    with pytest.raises(DagsterInvalidInvocationError, match="Decorated function 'op_requires_config' has context argument, but no context was provided when invoking."):
        op_requires_config()
    with pytest.raises(DagsterInvalidInvocationError, match="Decorated function 'op_requires_config' has context argument, but no context was provided when invoking."):
        op_requires_config.alias('aliased_op_requires_config')()
    with pytest.raises(DagsterInvalidConfigError, match='Error in config for op'):
        op_requires_config(None)
    context = build_op_context()
    with pytest.raises(DagsterInvalidConfigError, match='Error in config for op'):
        op_requires_config(context)
    with pytest.raises(DagsterInvalidInvocationError, match="Decorated function 'op_requires_config' has context argument, but no context was provided when invoking."):
        op_requires_config.configured({'foo': 'bar'}, name='configured_op')()
    result = op_requires_config.configured({'foo': 'bar'}, name='configured_op')(None)
    assert result == 5
    result = op_requires_config(build_op_context(op_config={'foo': 'bar'}))
    assert result == 5

def test_op_invocation_default_config():
    if False:
        return 10

    @op(config_schema={'foo': Field(str, is_required=False, default_value='bar')})
    def op_requires_config(context):
        if False:
            print('Hello World!')
        assert context.op_config['foo'] == 'bar'
        return context.op_config['foo']
    assert op_requires_config(None) == 'bar'

    @op(config_schema=Field(str, is_required=False, default_value='bar'))
    def op_requires_config_val(context):
        if False:
            print('Hello World!')
        assert context.op_config == 'bar'
        return context.op_config
    assert op_requires_config_val(None) == 'bar'

    @op(config_schema={'foo': Field(str, is_required=False, default_value='bar'), 'baz': str})
    def op_requires_config_partial(context):
        if False:
            while True:
                i = 10
        assert context.op_config['foo'] == 'bar'
        assert context.op_config['baz'] == 'bar'
        return context.op_config['foo'] + context.op_config['baz']
    assert op_requires_config_partial(build_op_context(op_config={'baz': 'bar'})) == 'barbar'

def test_op_invocation_dict_config():
    if False:
        return 10

    @op(config_schema=dict)
    def op_requires_dict(context):
        if False:
            return 10
        assert context.op_config == {'foo': 'bar'}
        return context.op_config
    assert op_requires_dict(build_op_context(op_config={'foo': 'bar'})) == {'foo': 'bar'}

    @op(config_schema=Noneable(dict))
    def op_noneable_dict(context):
        if False:
            print('Hello World!')
        return context.op_config
    assert op_noneable_dict(build_op_context()) is None
    assert op_noneable_dict(None) is None

def test_op_invocation_kitchen_sink_config():
    if False:
        i = 10
        return i + 15

    @op(config_schema={'str_field': str, 'int_field': int, 'list_int': [int], 'list_list_int': [[int]], 'dict_field': {'a_string': str}, 'list_dict_field': [{'an_int': int}], 'selector_of_things': Selector({'select_list_dict_field': [{'an_int': int}], 'select_int': int}), 'optional_list_of_optional_string': Noneable([Noneable(str)])})
    def kitchen_sink(context):
        if False:
            while True:
                i = 10
        return context.op_config
    op_config_one = {'str_field': 'kjf', 'int_field': 2, 'list_int': [3], 'list_list_int': [[1], [2, 3]], 'dict_field': {'a_string': 'kdjfkd'}, 'list_dict_field': [{'an_int': 2}, {'an_int': 4}], 'selector_of_things': {'select_int': 3}, 'optional_list_of_optional_string': ['foo', None]}
    assert kitchen_sink(build_op_context(op_config=op_config_one)) == op_config_one

def test_op_with_inputs():
    if False:
        i = 10
        return i + 15

    @op
    def op_with_inputs(x, y):
        if False:
            i = 10
            return i + 15
        assert x == 5
        assert y == 6
        return x + y
    assert op_with_inputs(5, 6) == 11
    assert op_with_inputs(x=5, y=6) == 11
    assert op_with_inputs(5, y=6) == 11
    assert op_with_inputs(y=6, x=5) == 11
    with pytest.raises(DagsterInvalidInvocationError, match='No value provided for required input "y".'):
        op_with_inputs(5)
    with pytest.raises(DagsterInvalidInvocationError, match="Too many input arguments were provided for op 'op_with_inputs'"):
        op_with_inputs(5, 6, 7)
    with pytest.raises(DagsterInvalidInvocationError, match="Too many input arguments were provided for op 'op_with_inputs'"):
        op_with_inputs(5, 6, z=7)
    with pytest.raises(DagsterInvalidInvocationError, match='No value provided for required input "y".'):
        op_with_inputs(5, z=5)

def test_failing_op():
    if False:
        for i in range(10):
            print('nop')

    @op
    def op_fails():
        if False:
            i = 10
            return i + 15
        raise Exception('Oh no!')
    with pytest.raises(Exception, match='Oh no!'):
        op_fails()

def test_attempted_invocation_in_composition():
    if False:
        i = 10
        return i + 15

    @op
    def basic_op(_x):
        if False:
            i = 10
            return i + 15
        pass
    msg = 'Must pass the output from previous node invocations or inputs to the composition function as inputs when invoking nodes during composition.'
    with pytest.raises(DagsterInvalidDefinitionError, match=msg):

        @job
        def _job_will_fail():
            if False:
                return 10
            basic_op(5)
    with pytest.raises(DagsterInvalidDefinitionError, match=msg):

        @job
        def _job_will_fail_again():
            if False:
                i = 10
                return i + 15
            basic_op(_x=5)

def test_async_op():
    if False:
        return 10

    @op
    async def aio_op():
        await asyncio.sleep(0.01)
        return 'done'
    assert asyncio.run(aio_op()) == 'done'

def test_async_gen_invocation():
    if False:
        for i in range(10):
            print('nop')

    async def make_outputs():
        await asyncio.sleep(0.01)
        yield Output('first', output_name='first')
        await asyncio.sleep(0.01)
        yield Output('second', output_name='second')

    @op(out={'first': Out(), 'second': Out()})
    async def aio_gen(_):
        async for v in make_outputs():
            yield v
    context = build_op_context()

    async def get_results():
        res = []
        async for output in aio_gen(context):
            res.append(output)
        return res
    results = asyncio.run(get_results())
    assert results[0].value == 'first'
    assert results[1].value == 'second'

    @graph
    def aio():
        if False:
            for i in range(10):
                print('nop')
        aio_gen()
    result = aio.execute_in_process()
    assert result.success
    assert result.output_for_node('aio_gen', 'first') == 'first'
    assert result.output_for_node('aio_gen', 'second') == 'second'

def test_multiple_outputs_iterator():
    if False:
        for i in range(10):
            print('nop')

    @op(out={'1': Out(int), '2': Out(int)})
    def op_multiple_outputs():
        if False:
            print('Hello World!')
        yield Output(2, output_name='2')
        yield Output(1, output_name='1')
    result = wrap_op_in_graph_and_execute(op_multiple_outputs)
    assert result.success
    outputs = list(op_multiple_outputs())
    assert outputs[0].value == 2
    assert outputs[1].value == 1

def test_wrong_output():
    if False:
        print('Hello World!')

    @op
    def op_wrong_output():
        if False:
            print('Hello World!')
        return Output(5, output_name='wrong_name')
    with pytest.raises(DagsterInvariantViolationError, match="explicitly named 'wrong_name'"):
        wrap_op_in_graph_and_execute(op_wrong_output)
    with pytest.raises(DagsterInvariantViolationError, match="explicitly named 'wrong_name'"):
        op_wrong_output()

def test_optional_output_return():
    if False:
        return 10

    @op(out={'1': Out(int, is_required=False), '2': Out(int)})
    def op_multiple_outputs_not_sent():
        if False:
            i = 10
            return i + 15
        return Output(2, output_name='2')
    with pytest.raises(DagsterInvariantViolationError, match='has multiple outputs, but only one output was returned'):
        op_multiple_outputs_not_sent()
    with pytest.raises(DagsterInvariantViolationError, match='has multiple outputs, but only one output was returned'):
        wrap_op_in_graph_and_execute(op_multiple_outputs_not_sent)

def test_optional_output_yielded():
    if False:
        print('Hello World!')

    @op(out={'1': Out(int, is_required=False), '2': Out(int)})
    def op_multiple_outputs_not_sent():
        if False:
            print('Hello World!')
        yield Output(2, output_name='2')
    assert next(iter(op_multiple_outputs_not_sent())).value == 2

def test_optional_output_yielded_async():
    if False:
        print('Hello World!')

    @op(out={'1': Out(int, is_required=False), '2': Out(int)})
    async def op_multiple_outputs_not_sent():
        yield Output(2, output_name='2')

    async def get_results():
        res = []
        async for output in op_multiple_outputs_not_sent():
            res.append(output)
        return res
    output = asyncio.run(get_results())[0]
    assert output.value == 2

def test_missing_required_output_generator():
    if False:
        return 10

    @op(out={'1': Out(int), '2': Out(int)})
    def op_multiple_outputs_not_sent():
        if False:
            print('Hello World!')
        yield Output(2, output_name='2')
    with pytest.raises(DagsterStepOutputNotFoundError, match='Core compute for op "op_multiple_outputs_not_sent" did not return an output for non-optional output "1"'):
        wrap_op_in_graph_and_execute(op_multiple_outputs_not_sent)
    with pytest.raises(DagsterInvariantViolationError, match='Invocation of op "op_multiple_outputs_not_sent" did not return an output for non-optional output "1"'):
        list(op_multiple_outputs_not_sent())

def test_missing_required_output_generator_async():
    if False:
        return 10

    @op(out={'1': Out(int), '2': Out(int)})
    async def op_multiple_outputs_not_sent():
        yield Output(2, output_name='2')
    with pytest.raises(DagsterStepOutputNotFoundError, match='Core compute for op "op_multiple_outputs_not_sent" did not return an output for non-optional output "1"'):
        wrap_op_in_graph_and_execute(op_multiple_outputs_not_sent)

    async def get_results():
        res = []
        async for output in op_multiple_outputs_not_sent():
            res.append(output)
        return res
    with pytest.raises(DagsterInvariantViolationError, match="Invocation of op 'op_multiple_outputs_not_sent' did not return an output for non-optional output '1'"):
        asyncio.run(get_results())

def test_missing_required_output_return():
    if False:
        print('Hello World!')

    @op(out={'1': Out(int), '2': Out(int)})
    def op_multiple_outputs_not_sent():
        if False:
            i = 10
            return i + 15
        return Output(2, output_name='2')
    with pytest.raises(DagsterInvariantViolationError, match='has multiple outputs, but only one output was returned'):
        wrap_op_in_graph_and_execute(op_multiple_outputs_not_sent)
    with pytest.raises(DagsterInvariantViolationError, match='has multiple outputs, but only one output was returned'):
        op_multiple_outputs_not_sent()

def test_output_sent_multiple_times():
    if False:
        while True:
            i = 10

    @op(out={'1': Out(int)})
    def op_yields_twice():
        if False:
            for i in range(10):
                print('nop')
        yield Output(1, '1')
        yield Output(2, '1')
    with pytest.raises(DagsterInvariantViolationError, match='Compute for op "op_yields_twice" returned an output "1" multiple times'):
        wrap_op_in_graph_and_execute(op_yields_twice)
    with pytest.raises(DagsterInvariantViolationError, match="Invocation of op 'op_yields_twice' yielded an output '1' multiple times"):
        list(op_yields_twice())
_invalid_on_bound = [('dagster_run', None), ('step_launcher', None), ('job_def', None), ('job_name', None), ('node_handle', None), ('op', None), ('get_step_execution_context', None)]

@pytest.mark.parametrize('property_or_method_name,val_to_pass', _invalid_on_bound)
def test_invalid_properties_on_bound_context(property_or_method_name: str, val_to_pass: object):
    if False:
        return 10

    @op
    def op_fails_getting_property(context):
        if False:
            return 10
        result = getattr(context, property_or_method_name)
        result(val_to_pass) if val_to_pass else result()
    with pytest.raises(DagsterInvalidPropertyError):
        op_fails_getting_property(build_op_context())

def test_bound_context():
    if False:
        while True:
            i = 10

    @op
    def access_bound_details(context: OpExecutionContext):
        if False:
            print('Hello World!')
        assert context.op_def
    access_bound_details(build_op_context())

@pytest.mark.parametrize('property_or_method_name,val_to_pass', [*_invalid_on_bound, ('op_def', None), ('assets_def', None)])
def test_invalid_properties_on_unbound_context(property_or_method_name: str, val_to_pass: object):
    if False:
        return 10
    context = build_op_context()
    with pytest.raises(DagsterInvalidPropertyError):
        result = getattr(context, property_or_method_name)
        result(val_to_pass) if val_to_pass else result()

def test_op_retry_requested():
    if False:
        return 10

    @op
    def op_retries():
        if False:
            return 10
        raise RetryRequested()
    with pytest.raises(RetryRequested):
        op_retries()

def test_op_failure():
    if False:
        i = 10
        return i + 15

    @op
    def op_fails():
        if False:
            i = 10
            return i + 15
        raise Failure('oops')
    with pytest.raises(Failure, match='oops'):
        op_fails()

def test_yielded_asset_materialization():
    if False:
        return 10

    @op
    def op_yields_materialization(_):
        if False:
            for i in range(10):
                print('nop')
        yield AssetMaterialization(asset_key=AssetKey(['fake']))
        yield Output(5)
        yield AssetMaterialization(asset_key=AssetKey(['fake2']))
    events = list(op_yields_materialization(None))
    outputs = [event for event in events if isinstance(event, Output)]
    assert outputs[0].value == 5
    materializations = [materialization for materialization in events if isinstance(materialization, AssetMaterialization)]
    assert len(materializations) == 2

def test_input_type_check():
    if False:
        i = 10
        return i + 15

    @op(ins={'x': In(dagster_type=int)})
    def op_takes_input(x):
        if False:
            while True:
                i = 10
        return x + 1
    assert op_takes_input(5) == 6
    with pytest.raises(DagsterTypeCheckDidNotPass, match='Description: Value "foo" of python type "str" must be a int.'):
        op_takes_input('foo')

def test_output_type_check():
    if False:
        while True:
            i = 10

    @op(out=Out(dagster_type=int))
    def wrong_type():
        if False:
            for i in range(10):
                print('nop')
        return 'foo'
    with pytest.raises(DagsterTypeCheckDidNotPass, match='Description: Value "foo" of python type "str" must be a int.'):
        wrong_type()

def test_pending_node_invocation():
    if False:
        i = 10
        return i + 15

    @op
    def basic_op_to_hook():
        if False:
            print('Hello World!')
        return 5
    assert basic_op_to_hook.with_hooks(set())() == 5

    @op
    def basic_op_with_tag(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.has_tag('foo')
        return context.get_tag('foo')
    assert basic_op_with_tag.tag({'foo': 'bar'})(None) == 'bar'

def test_graph_invocation_out_of_composition():
    if False:
        i = 10
        return i + 15

    @op
    def basic_op():
        if False:
            print('Hello World!')
        return 5

    @graph
    def the_graph():
        if False:
            print('Hello World!')
        basic_op()
    with pytest.raises(DagsterInvariantViolationError, match="Attempted to call graph 'the_graph' outside of a composition function. Invoking graphs is only valid in a function decorated with @job or @graph."):
        the_graph()

def test_job_invocation():
    if False:
        return 10

    @job
    def basic_job():
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(DagsterInvariantViolationError, match="Attempted to call job 'basic_job' directly. Jobs should be invoked by using an execution API function \\(e.g. `job.execute_in_process`\\)."):
        basic_job()

@op
async def foo_async() -> str:
    return 'bar'

def test_coroutine_asyncio_invocation():
    if False:
        i = 10
        return i + 15

    async def my_coroutine_test():
        result = await foo_async()
        assert result == 'bar'
    asyncio.run(my_coroutine_test())

def test_op_invocation_nothing_deps():
    if False:
        i = 10
        return i + 15

    @op(ins={'start': In(Nothing)})
    def nothing_dep():
        if False:
            for i in range(10):
                print('nop')
        return 5
    with pytest.raises(DagsterInvalidInvocationError, match="Attempted to provide value for nothing input 'start'. Nothing dependencies are ignored when directly invoking ops."):
        nothing_dep(start='blah')
    with pytest.raises(DagsterInvalidInvocationError, match="Too many input arguments were provided for op 'nothing_dep'. This may be because you attempted to provide a value for a nothing dependency. Nothing dependencies are ignored when directly invoking ops."):
        nothing_dep('blah')
    assert nothing_dep() == 5

    @op(ins={'x': In(), 'y': In(Nothing), 'z': In()})
    def sandwiched_nothing_dep(x, z):
        if False:
            while True:
                i = 10
        return x + z
    assert sandwiched_nothing_dep(5, 6) == 11
    with pytest.raises(DagsterInvalidInvocationError, match="Too many input arguments were provided for op 'sandwiched_nothing_dep'. This may be because you attempted to provide a value for a nothing dependency. Nothing dependencies are ignored when directly invoking ops."):
        sandwiched_nothing_dep(5, 6, 7)

def test_dynamic_output_gen():
    if False:
        for i in range(10):
            print('nop')

    @op(out={'a': DynamicOut(is_required=False), 'b': Out(is_required=False)})
    def my_dynamic():
        if False:
            for i in range(10):
                print('nop')
        yield DynamicOutput(value=1, mapping_key='1', output_name='a')
        yield DynamicOutput(value=2, mapping_key='2', output_name='a')
        yield Output(value='foo', output_name='b')
    (a1, a2, b) = my_dynamic()
    assert a1.value == 1
    assert a1.mapping_key == '1'
    assert a2.value == 2
    assert a2.mapping_key == '2'
    assert b.value == 'foo'

def test_dynamic_output_async_gen():
    if False:
        while True:
            i = 10

    @op(out={'a': DynamicOut(is_required=False), 'b': Out(is_required=False)})
    async def aio_gen():
        yield DynamicOutput(value=1, mapping_key='1', output_name='a')
        yield DynamicOutput(value=2, mapping_key='2', output_name='a')
        await asyncio.sleep(0.01)
        yield Output(value='foo', output_name='b')

    async def get_results():
        res = []
        async for output in aio_gen():
            res.append(output)
        return res
    (a1, a2, b) = asyncio.run(get_results())
    assert a1.value == 1
    assert a1.mapping_key == '1'
    assert a2.value == 2
    assert a2.mapping_key == '2'
    assert b.value == 'foo'

def test_dynamic_output_non_gen():
    if False:
        print('Hello World!')

    @op(out={'a': DynamicOut(is_required=False)})
    def should_not_work():
        if False:
            i = 10
            return i + 15
        return DynamicOutput(value=1, mapping_key='1', output_name='a')
    with pytest.raises(DagsterInvariantViolationError, match='expected a list of DynamicOutput objects'):
        should_not_work()
    with pytest.raises(DagsterInvariantViolationError, match='expected a list of DynamicOutput objects'):
        wrap_op_in_graph_and_execute(should_not_work)

def test_dynamic_output_async_non_gen():
    if False:
        i = 10
        return i + 15

    @op(out={'a': DynamicOut(is_required=False)})
    async def should_not_work():
        await asyncio.sleep(0.01)
        return DynamicOutput(value=1, mapping_key='1', output_name='a')
    with pytest.raises(DagsterInvariantViolationError, match="dynamic output 'a' expected a list of DynamicOutput objects"):
        asyncio.run(should_not_work())
    with pytest.raises(Exception):
        wrap_op_in_graph_and_execute(should_not_work())

def test_op_invocation_with_bad_resources(capsys):
    if False:
        for i in range(10):
            print('nop')

    @resource
    def bad_resource(_):
        if False:
            for i in range(10):
                print('nop')
        if 1 == 1:
            raise Exception('oopsy daisy')
        yield 'foo'

    @op(required_resource_keys={'my_resource'})
    def op_requires_resource(context):
        if False:
            return 10
        return context.resources.my_resource
    with pytest.raises(DagsterResourceFunctionError, match='Error executing resource_fn on ResourceDefinition my_resource'):
        with build_op_context(resources={'my_resource': bad_resource}) as context:
            assert op_requires_resource(context) == 'foo'
    captured = capsys.readouterr()
    assert 'Exception ignored in' not in captured.err

@pytest.mark.parametrize('context_builder', [build_op_context, build_op_context])
def test_build_context_with_resources_config(context_builder):
    if False:
        i = 10
        return i + 15

    @resource(config_schema=str)
    def my_resource(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.resource_config == 'foo'

    @op(required_resource_keys={'my_resource'})
    def my_op(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.run_config['resources']['my_resource'] == {'config': 'foo'}
    context = context_builder(resources={'my_resource': my_resource}, resources_config={'my_resource': {'config': 'foo'}})
    my_op(context)
    with pytest.raises(DagsterInvalidConfigError, match='Received unexpected config entry "bad_resource" at the root.'):
        context_builder(resources={'my_resource': my_resource}, resources_config={'bad_resource': {'config': 'foo'}})

def test_logged_user_events():
    if False:
        i = 10
        return i + 15

    @op
    def logs_events(context):
        if False:
            for i in range(10):
                print('nop')
        context.log_event(AssetMaterialization('first'))
        context.log_event(ExpectationResult(success=True))
        context.log_event(AssetObservation('fourth'))
        yield AssetMaterialization('fifth')
        yield Output('blah')
    context = build_op_context()
    list(logs_events(context))
    assert [type(event) for event in context.get_events()] == [AssetMaterialization, ExpectationResult, AssetObservation]

def test_add_output_metadata():
    if False:
        for i in range(10):
            print('nop')

    @op(out={'out1': Out(), 'out2': Out()})
    def the_op(context):
        if False:
            print('Hello World!')
        context.add_output_metadata({'foo': 'bar'}, output_name='out1')
        yield Output(value=1, output_name='out1')
        context.add_output_metadata({'bar': 'baz'}, output_name='out2')
        yield Output(value=2, output_name='out2')
    context = build_op_context()
    events = list(the_op(context))
    assert len(events) == 2
    assert context.get_output_metadata('out1') == {'foo': 'bar'}
    assert context.get_output_metadata('out2') == {'bar': 'baz'}

def test_add_output_metadata_after_output():
    if False:
        print('Hello World!')

    @op
    def the_op(context):
        if False:
            while True:
                i = 10
        yield Output(value=1)
        context.add_output_metadata({'foo': 'bar'})
    with pytest.raises(DagsterInvariantViolationError, match="In op 'the_op', attempted to log output metadata for output 'result' which has already been yielded. Metadata must be logged before the output is yielded."):
        list(the_op(build_op_context()))

def test_log_metadata_multiple_dynamic_outputs():
    if False:
        return 10

    @op(out={'out1': DynamicOut(), 'out2': DynamicOut()})
    def the_op(context):
        if False:
            print('Hello World!')
        context.add_output_metadata({'one': 'one'}, output_name='out1', mapping_key='one')
        yield DynamicOutput(value=1, output_name='out1', mapping_key='one')
        context.add_output_metadata({'two': 'two'}, output_name='out1', mapping_key='two')
        context.add_output_metadata({'three': 'three'}, output_name='out2', mapping_key='three')
        yield DynamicOutput(value=2, output_name='out1', mapping_key='two')
        yield DynamicOutput(value=3, output_name='out2', mapping_key='three')
        context.add_output_metadata({'four': 'four'}, output_name='out2', mapping_key='four')
        yield DynamicOutput(value=4, output_name='out2', mapping_key='four')
    context = build_op_context()
    events = list(the_op(context))
    assert len(events) == 4
    assert context.get_output_metadata('out1', mapping_key='one') == {'one': 'one'}
    assert context.get_output_metadata('out1', mapping_key='two') == {'two': 'two'}
    assert context.get_output_metadata('out2', mapping_key='three') == {'three': 'three'}
    assert context.get_output_metadata('out2', mapping_key='four') == {'four': 'four'}

def test_log_metadata_after_dynamic_output():
    if False:
        i = 10
        return i + 15

    @op(out=DynamicOut())
    def the_op(context):
        if False:
            while True:
                i = 10
        yield DynamicOutput(1, mapping_key='one')
        context.add_output_metadata({'foo': 'bar'}, mapping_key='one')
    with pytest.raises(DagsterInvariantViolationError, match="In op 'the_op', attempted to log output metadata for output 'result' with mapping_key 'one' which has already been yielded. Metadata must be logged before the output is yielded."):
        list(the_op(build_op_context()))

def test_kwarg_inputs():
    if False:
        i = 10
        return i + 15

    @op(ins={'the_in': In(str)})
    def the_op(**kwargs) -> str:
        if False:
            print('Hello World!')
        return kwargs['the_in'] + 'foo'
    with pytest.raises(DagsterInvalidInvocationError, match="'the_op' has 0 positional inputs, but 1 positional inputs were provided."):
        the_op('bar')
    assert the_op(the_in='bar') == 'barfoo'
    with pytest.raises(KeyError):
        the_op(bad_val='bar')

    @op(ins={'the_in': In(), 'kwarg_in': In(), 'kwarg_in_two': In()})
    def the_op_2(the_in, **kwargs):
        if False:
            while True:
                i = 10
        return the_in + kwargs['kwarg_in'] + kwargs['kwarg_in_two']
    assert the_op_2('foo', kwarg_in='bar', kwarg_in_two='baz') == 'foobarbaz'

def test_kwarg_inputs_context():
    if False:
        print('Hello World!')
    context = build_op_context()

    @op(ins={'the_in': In(str)})
    def the_op(context, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        assert context
        return kwargs['the_in'] + 'foo'
    with pytest.raises(DagsterInvalidInvocationError, match="'the_op' has 0 positional inputs, but 1 positional inputs were provided."):
        the_op(context, 'bar')
    assert the_op(context, the_in='bar') == 'barfoo'
    with pytest.raises(KeyError):
        the_op(context, bad_val='bar')

    @op(ins={'the_in': In(), 'kwarg_in': In(), 'kwarg_in_two': In()})
    def the_op_2(context, the_in, **kwargs):
        if False:
            i = 10
            return i + 15
        assert context
        return the_in + kwargs['kwarg_in'] + kwargs['kwarg_in_two']
    assert the_op_2(context, 'foo', kwarg_in='bar', kwarg_in_two='baz') == 'foobarbaz'

def test_default_kwarg_inputs():
    if False:
        while True:
            i = 10

    @op
    def the_op(x=1, y=2):
        if False:
            i = 10
            return i + 15
        return x + y
    assert the_op() == 3

def test_kwargs_via_partial_functools():
    if False:
        i = 10
        return i + 15

    def fake_func(foo, bar):
        if False:
            print('Hello World!')
        return foo + bar
    new_func = partial(fake_func, foo=1, bar=2)
    new_op = op(name='new_func')(new_func)
    assert new_op() == 3

def test_get_mapping_key():
    if False:
        while True:
            i = 10
    context = build_op_context(mapping_key='the_key')
    assert context.get_mapping_key() == 'the_key'

    @op
    def basic_op(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.get_mapping_key() == 'the_key'
    basic_op(context)

def test_required_resource_keys_no_context_invocation():
    if False:
        i = 10
        return i + 15

    @op(required_resource_keys={'foo'})
    def uses_resource_no_context():
        if False:
            print('Hello World!')
        pass
    uses_resource_no_context()
    with pytest.raises(DagsterInvalidInvocationError, match="Too many input arguments were provided for op 'uses_resource_no_context'. This may be because an argument was provided for the context parameter, but no context parameter was defined for the op."):
        uses_resource_no_context(None)

def test_assets_def_invocation():
    if False:
        for i in range(10):
            print('nop')

    @asset()
    def my_asset(context):
        if False:
            print('Hello World!')
        assert context.assets_def == my_asset

    @op
    def non_asset_op(context):
        if False:
            print('Hello World!')
        context.assets_def
    with build_op_context(partition_key='2023-02-02') as context:
        my_asset(context)
        with pytest.raises(DagsterInvalidPropertyError, match='does not have an assets definition'):
            non_asset_op(context)

def test_partitions_time_window_asset_invocation():
    if False:
        print('Hello World!')
    partitions_def = DailyPartitionsDefinition(start_date=datetime(2023, 1, 1))

    @asset(partitions_def=partitions_def)
    def partitioned_asset(context):
        if False:
            return 10
        (start, end) = context.asset_partitions_time_window_for_output()
        assert start == pendulum.instance(datetime(2023, 2, 2), tz=partitions_def.timezone)
        assert end == pendulum.instance(datetime(2023, 2, 3), tz=partitions_def.timezone)
    context = build_op_context(partition_key='2023-02-02')
    partitioned_asset(context)

def test_multipartitioned_time_window_asset_invocation():
    if False:
        print('Hello World!')
    partitions_def = MultiPartitionsDefinition({'date': DailyPartitionsDefinition(start_date='2020-01-01'), 'static': StaticPartitionsDefinition(['a', 'b'])})

    @asset(partitions_def=partitions_def)
    def my_asset(context):
        if False:
            for i in range(10):
                print('nop')
        time_window = TimeWindow(start=pendulum.instance(datetime(year=2020, month=1, day=1), tz=get_time_partitions_def(partitions_def).timezone), end=pendulum.instance(datetime(year=2020, month=1, day=2), tz=get_time_partitions_def(partitions_def).timezone))
        assert context.asset_partitions_time_window_for_output() == time_window
        return 1
    context = build_op_context(partition_key='2020-01-01|a')
    my_asset(context)
    partitions_def = MultiPartitionsDefinition({'static2': StaticPartitionsDefinition(['a', 'b']), 'static': StaticPartitionsDefinition(['a', 'b'])})

    @asset(partitions_def=partitions_def)
    def static_multipartitioned_asset(context):
        if False:
            while True:
                i = 10
        with pytest.raises(DagsterInvariantViolationError, match='with a single time dimension'):
            context.asset_partitions_time_window_for_output()
    context = build_op_context(partition_key='a|a')
    static_multipartitioned_asset(context)

def test_partition_range_asset_invocation():
    if False:
        print('Hello World!')
    partitions_def = DailyPartitionsDefinition(start_date=datetime(2023, 1, 1))

    @asset(partitions_def=partitions_def)
    def foo(context: AssetExecutionContext):
        if False:
            while True:
                i = 10
        keys = partitions_def.get_partition_keys_in_range(context.partition_key_range)
        return {k: True for k in keys}
    context = build_op_context(partition_key_range=PartitionKeyRange('2023-01-01', '2023-01-02'))
    assert foo(context) == {'2023-01-01': True, '2023-01-02': True}
    context = build_asset_context(partition_key_range=PartitionKeyRange('2023-01-01', '2023-01-02'))
    assert foo(context) == {'2023-01-01': True, '2023-01-02': True}