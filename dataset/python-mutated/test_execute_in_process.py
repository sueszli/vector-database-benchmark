import pytest
from dagster import AssetKey, AssetMaterialization, AssetObservation, DagsterInvariantViolationError, DynamicOut, DynamicOutput, Out, Output, RetryRequested, daily_partitioned_config, graph, job, mem_io_manager, op, resource
from dagster._core.definitions.output import GraphOut
from dagster._core.errors import DagsterMaxRetriesExceededError

def get_solids():
    if False:
        while True:
            i = 10

    @op
    def emit_one():
        if False:
            while True:
                i = 10
        return 1

    @op
    def add(x, y):
        if False:
            return 10
        return x + y
    return (emit_one, add)

def test_output_value():
    if False:
        while True:
            i = 10

    @graph
    def a():
        if False:
            return 10
        get_solids()[0]()
    result = a.execute_in_process()
    assert result.success
    assert result.output_for_node('emit_one') == 1

def test_output_values():
    if False:
        while True:
            i = 10

    @op(out={'a': Out(), 'b': Out()})
    def two_outs():
        if False:
            print('Hello World!')
        return (1, 2)

    @graph
    def a():
        if False:
            print('Hello World!')
        two_outs()
    result = a.execute_in_process()
    assert result.success
    assert result.output_for_node('two_outs', 'a') == 1
    assert result.output_for_node('two_outs', 'b') == 2

def test_dynamic_output_values():
    if False:
        for i in range(10):
            print('nop')

    @op(out=DynamicOut())
    def two_outs():
        if False:
            i = 10
            return i + 15
        yield DynamicOutput(1, 'a')
        yield DynamicOutput(2, 'b')

    @op
    def add_one(x):
        if False:
            return 10
        return x + 1

    @graph
    def a():
        if False:
            while True:
                i = 10
        two_outs().map(add_one)
    result = a.execute_in_process()
    assert result.success
    assert result.output_for_node('two_outs') == {'a': 1, 'b': 2}
    assert result.output_for_node('add_one') == {'a': 2, 'b': 3}

def test_execute_graph():
    if False:
        return 10
    (emit_one, add) = get_solids()

    @graph
    def emit_two():
        if False:
            print('Hello World!')
        return add(emit_one(), emit_one())

    @graph
    def emit_three():
        if False:
            print('Hello World!')
        return add(emit_two(), emit_one())
    result = emit_three.execute_in_process()
    assert result.success
    assert result.output_value() == 3
    assert result.output_for_node('add') == 3
    assert result.output_for_node('emit_two') == 2
    assert result.output_for_node('emit_one') == 1
    assert result.output_for_node('emit_two.emit_one') == 1
    assert result.output_for_node('emit_two.emit_one_2') == 1

def test_graph_with_required_resources():
    if False:
        print('Hello World!')

    @op(required_resource_keys={'a'})
    def basic_reqs(context):
        if False:
            i = 10
            return i + 15
        return context.resources.a

    @graph
    def basic_graph():
        if False:
            while True:
                i = 10
        return basic_reqs()
    result = basic_graph.execute_in_process(resources={'a': 'foo'})
    assert result.output_value() == 'foo'

    @resource
    def basic_resource():
        if False:
            i = 10
            return i + 15
        return 'bar'
    result = basic_graph.execute_in_process(resources={'a': basic_resource})
    assert result.output_value() == 'bar'

def test_executor_config_ignored_by_execute_in_process():
    if False:
        i = 10
        return i + 15

    @op
    def my_op():
        if False:
            return 10
        return 0

    @graph
    def my_graph():
        if False:
            return 10
        my_op()
    my_job = my_graph.to_job(config={'execution': {'config': {'multiprocess': {'max_concurrent': 5}}}})
    result = my_job.execute_in_process()
    assert result.success

def test_output_for_node_composite():
    if False:
        return 10

    @op(out={'foo': Out()})
    def my_op():
        if False:
            while True:
                i = 10
        return 5

    @graph(out={'bar': GraphOut()})
    def my_graph():
        if False:
            print('Hello World!')
        return my_op()

    @graph(out={'baz': GraphOut()})
    def my_top_graph():
        if False:
            i = 10
            return i + 15
        return my_graph()
    result = my_graph.execute_in_process()
    assert result.success
    assert result.output_for_node('my_op', 'foo') == 5
    assert result.output_value('bar') == 5
    result = my_top_graph.execute_in_process()
    assert result.output_for_node('my_graph', 'bar') == 5
    assert result.output_for_node('my_graph.my_op', 'foo') == 5
    assert result.output_value('baz') == 5

def test_output_for_node_not_found():
    if False:
        for i in range(10):
            print('nop')

    @op
    def op_exists():
        if False:
            return 10
        return 5

    @graph
    def basic():
        if False:
            print('Hello World!')
        return op_exists()
    result = basic.execute_in_process()
    assert result.success
    with pytest.raises(DagsterInvariantViolationError, match='name_doesnt_exist'):
        result.output_for_node('op_exists', 'name_doesnt_exist')
    with pytest.raises(DagsterInvariantViolationError, match="Could not find top-level output 'name_doesnt_exist'"):
        result.output_value('name_doesnt_exist')
    with pytest.raises(DagsterInvariantViolationError, match='basic has no op named op_doesnt_exist'):
        result.output_for_node('op_doesnt_exist')

def _get_step_successes(event_list):
    if False:
        for i in range(10):
            print('nop')
    return [event for event in event_list if event.is_step_success]

def test_step_events_for_node():
    if False:
        i = 10
        return i + 15

    @op
    def op_exists():
        if False:
            for i in range(10):
                print('nop')
        return 5

    @graph
    def basic():
        if False:
            i = 10
            return i + 15
        return op_exists()

    @graph
    def nested():
        if False:
            for i in range(10):
                print('nop')
        return basic()
    result = nested.execute_in_process()
    node_events = result.all_node_events
    assert len(_get_step_successes(node_events)) == 1
    basic_events = result.events_for_node('basic')
    assert len(_get_step_successes(basic_events)) == 1
    op_events = result.events_for_node('basic.op_exists')
    assert len(_get_step_successes(op_events)) == 1

def test_output_value_error():
    if False:
        while True:
            i = 10

    @job
    def my_job():
        if False:
            print('Hello World!')
        pass
    result = my_job.execute_in_process()
    with pytest.raises(DagsterInvariantViolationError, match="Attempted to retrieve top-level outputs for 'my_job', which has no outputs."):
        result.output_value()

def test_partitions_key():
    if False:
        for i in range(10):
            print('nop')

    @op
    def my_op(context):
        if False:
            for i in range(10):
                print('nop')
        assert context._step_execution_context.plan_data.dagster_run.tags['dagster/partition'] == '2020-01-01'

    @daily_partitioned_config(start_date='2020-01-01')
    def my_partitioned_config(_start, _end):
        if False:
            for i in range(10):
                print('nop')
        return {}

    @job(config=my_partitioned_config)
    def my_job():
        if False:
            print('Hello World!')
        my_op()
    assert my_job.execute_in_process(partition_key='2020-01-01').success

def test_asset_materialization():
    if False:
        i = 10
        return i + 15

    @op(out={})
    def my_op():
        if False:
            i = 10
            return i + 15
        yield AssetMaterialization('abc')

    @job
    def my_job():
        if False:
            print('Hello World!')
        my_op()
    result = my_job.execute_in_process()
    assert result.asset_materializations_for_node('my_op') == [AssetMaterialization(asset_key=AssetKey(['abc']))]

def test_asset_observation():
    if False:
        for i in range(10):
            print('nop')

    @op(out={})
    def my_op():
        if False:
            print('Hello World!')
        yield AssetObservation('abc')

    @job
    def my_job():
        if False:
            i = 10
            return i + 15
        my_op()
    result = my_job.execute_in_process()
    assert result.asset_observations_for_node('my_op') == [AssetObservation(asset_key=AssetKey(['abc']))]

def test_dagster_run():
    if False:
        for i in range(10):
            print('nop')

    @op
    def success_op():
        if False:
            print('Hello World!')
        return True

    @job
    def my_success_job():
        if False:
            print('Hello World!')
        success_op()
    result = my_success_job.execute_in_process()
    assert result.success
    assert result.dagster_run.is_success

    @op
    def fail_op():
        if False:
            while True:
                i = 10
        raise Exception

    @job
    def my_failure_job():
        if False:
            return 10
        fail_op()
    result = my_failure_job.execute_in_process(raise_on_error=False)
    assert not result.success
    assert not result.dagster_run.is_success

def test_dynamic_output_for_node():
    if False:
        i = 10
        return i + 15

    @op(out=DynamicOut())
    def fanout():
        if False:
            for i in range(10):
                print('nop')
        for i in range(3):
            yield DynamicOutput(value=i, mapping_key=str(i))

    @op(out={'output1': Out(int), 'output2': Out(int)})
    def return_as_tuple(x):
        if False:
            for i in range(10):
                print('nop')
        yield Output(value=x, output_name='output1')
        yield Output(value=5, output_name='output2')

    @job
    def myjob():
        if False:
            return 10
        fanout().map(return_as_tuple)
    result = myjob.execute_in_process()
    assert result.output_for_node('return_as_tuple', 'output1') == {'0': 0, '1': 1, '2': 2}
    assert result.output_for_node('return_as_tuple', 'output2') == {'0': 5, '1': 5, '2': 5}

def test_execute_in_process_input_values():
    if False:
        print('Hello World!')

    @op
    def requires_input_op(x: int):
        if False:
            i = 10
            return i + 15
        return x + 1

    @graph
    def requires_input_graph(x):
        if False:
            i = 10
            return i + 15
        return requires_input_op(x)
    result = requires_input_graph.alias('named_graph').execute_in_process(input_values={'x': 5})
    assert result.success
    assert result.output_value() == 6
    result = requires_input_graph.to_job().execute_in_process(input_values={'x': 5})
    assert result.success
    assert result.output_value() == 6

def test_retries_exceeded():
    if False:
        while True:
            i = 10
    called = []

    @op
    def always_fail():
        if False:
            while True:
                i = 10
        exception = Exception('I have failed.')
        called.append('yes')
        raise RetryRequested(max_retries=2) from exception

    @graph
    def fail():
        if False:
            for i in range(10):
                print('nop')
        always_fail()
    with pytest.raises(DagsterMaxRetriesExceededError, match='Exceeded max_retries of 2'):
        fail.execute_in_process()
    result = fail.execute_in_process(raise_on_error=False)
    assert not result.success
    assert 'Exception: I have failed' in result.filter_events(lambda evt: evt.is_step_failure)[0].event_specific_data.error_display_string

def test_execute_in_process_defaults_override():
    if False:
        print('Hello World!')

    @op
    def some_op(context):
        if False:
            while True:
                i = 10
        assert context.job_def.resource_defs['io_manager'] == mem_io_manager

    @graph
    def some_graph():
        if False:
            for i in range(10):
                print('nop')
        some_op()
    some_graph.execute_in_process()
    some_graph.to_job().execute_in_process()
    some_graph.alias('hello').execute_in_process()