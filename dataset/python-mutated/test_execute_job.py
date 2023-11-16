import pytest
from dagster import AssetKey, DagsterExecutionStepNotFoundError, DagsterInvalidConfigError, DagsterInvariantViolationError, Field, OpExecutionContext, Out, Output, ReexecutionOptions, asset, define_asset_job, execute_job, graph, in_process_executor, job, op, reconstructable, repository
from dagster._core.test_utils import instance_for_test

@pytest.fixture(scope='session')
def instance():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        yield instance

def emit_job():
    if False:
        return 10

    @op
    def emit_five():
        if False:
            for i in range(10):
                print('nop')
        return 5

    @op
    def returns_six(x):
        if False:
            print('Hello World!')
        return x + 1

    @graph
    def nested():
        if False:
            for i in range(10):
                print('nop')
        return returns_six(emit_five())

    @op(config_schema={'baz': Field(str, default_value='blah')})
    def conditional_return(context, x):
        if False:
            return 10
        if context.op_config['baz'] == 'blah':
            return x + 1
        else:
            return x + 2

    @job
    def the_job():
        if False:
            i = 10
            return i + 15
        conditional_return(nested())
    return the_job

def emit_error_job():
    if False:
        i = 10
        return i + 15

    @op
    def the_op_fails():
        if False:
            i = 10
            return i + 15
        raise Exception()

    @job
    def the_job_fails():
        if False:
            print('Hello World!')
        the_op_fails()
    return the_job_fails

def test_basic_success(instance):
    if False:
        i = 10
        return i + 15
    result = execute_job(reconstructable(emit_job), instance)
    assert result.success

def test_no_raise_on_error(instance):
    if False:
        print('Hello World!')
    result = execute_job(reconstructable(emit_error_job), instance)
    assert not result.success

def test_tags_for_run(instance):
    if False:
        print('Hello World!')
    result = execute_job(reconstructable(emit_job), instance, tags={'foo': 'bar'})
    assert result.success
    run = instance.get_run_by_id(result.run_id)
    assert run.tags == {'foo': 'bar'}

def test_run_config(instance):
    if False:
        while True:
            i = 10
    with execute_job(reconstructable(emit_job), instance, run_config={'ops': {'conditional_return': {'config': {'baz': 'not_blah'}}}}) as result:
        assert result.success
        assert result.output_for_node('conditional_return') == 8
    with pytest.raises(DagsterInvalidConfigError):
        execute_job(reconstructable(emit_job), instance, run_config={'ops': {'conditional_return': 'bad_config'}}, raise_on_error=True)

def test_retrieve_outputs_not_context_manager(instance):
    if False:
        return 10
    result = execute_job(reconstructable(emit_job), instance)
    with pytest.raises(DagsterInvariantViolationError, match='must be opened as a context manager'):
        result.output_for_node('nested')

def test_op_selection(instance):
    if False:
        print('Hello World!')
    with execute_job(reconstructable(emit_job), instance, op_selection=['nested.returns_six'], run_config={'ops': {'nested': {'ops': {'returns_six': {'inputs': {'x': {'value': 5}}}}}}}) as result:
        assert result.success
        assert result.output_for_node('nested.returns_six') == 6
        with pytest.raises(DagsterInvariantViolationError):
            result.output_for_node('conditional_return')

def test_result_output_access(instance):
    if False:
        i = 10
        return i + 15
    result = execute_job(reconstructable(emit_job), instance)
    with result:
        assert result.output_for_node('conditional_return') == 7
    with pytest.raises(DagsterInvariantViolationError):
        result.output_for_node('conditional_return')

def emit_based_on_config():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema=str)
    def the_op(context):
        if False:
            i = 10
            return i + 15
        return context.op_config

    @op
    def ingest(x):
        if False:
            for i in range(10):
                print('nop')
        return x

    @job
    def the_job():
        if False:
            return 10
        ingest(the_op())
    return the_job

def test_reexecution_with_steps(instance):
    if False:
        while True:
            i = 10
    with execute_job(reconstructable(emit_based_on_config), instance, run_config={'ops': {'the_op': {'config': 'blah'}}}) as result:
        assert result.success
        assert result.output_for_node('ingest') == 'blah'
    reexecution_options = ReexecutionOptions(parent_run_id=result.run_id, step_selection=['ingest'])
    with execute_job(reconstructable(emit_based_on_config), instance, reexecution_options=reexecution_options) as result:
        assert result.success
        assert result.output_for_node('ingest') == 'blah'
        assert len(result.get_step_success_events()) == 1

def error_on_config():
    if False:
        return 10

    @op
    def start():
        if False:
            while True:
                i = 10
        return 5

    @op(config_schema=str)
    def the_op_errors(context, x):
        if False:
            while True:
                i = 10
        if context.op_config == 'blah':
            raise Exception()
        else:
            return x

    @job
    def the_job():
        if False:
            print('Hello World!')
        the_op_errors(start())
    return the_job

def test_reexecution_from_failure(instance):
    if False:
        print('Hello World!')
    with execute_job(reconstructable(error_on_config), instance, run_config={'ops': {'the_op_errors': {'config': 'blah'}}}) as result:
        assert not result.success
    reexecution_options = ReexecutionOptions.from_failure(result.run_id, instance)
    with execute_job(reconstructable(error_on_config), instance, run_config={'ops': {'the_op_errors': {'config': 'no'}}}, reexecution_options=reexecution_options) as result:
        assert result.success
        assert result.output_for_node('the_op_errors') == 5
        assert len(result.get_step_success_events()) == 1

def test_reexecution_steps_dont_match(instance):
    if False:
        return 10
    with execute_job(reconstructable(emit_job), instance, op_selection=['conditional_return'], run_config={'ops': {'conditional_return': {'inputs': {'x': {'value': 4}}}}}) as result:
        assert result.success
        assert result.output_for_node('conditional_return') == 5
    reexecution_options = ReexecutionOptions(result.run_id, step_selection=['nested.returns_six', 'nested.emit_five'])
    with pytest.raises(DagsterExecutionStepNotFoundError, match='unknown steps'):
        execute_job(reconstructable(emit_job), instance, reexecution_options=reexecution_options)
    with execute_job(reconstructable(emit_job), instance) as result:
        assert result.success
        assert result.output_for_node('conditional_return') == 7
    reexecution_options = ReexecutionOptions(result.run_id, step_selection=['nested.returns_six', 'nested.emit_five'])
    with execute_job(reconstructable(emit_job), instance, reexecution_options=reexecution_options) as result:
        assert result.success
        assert result.output_for_node('nested') == 6
        assert len(result.get_step_success_events()) == 2

def test_reexecute_from_failure_successful_run(instance):
    if False:
        while True:
            i = 10
    with execute_job(reconstructable(emit_job), instance, op_selection=['conditional_return'], run_config={'ops': {'conditional_return': {'inputs': {'x': {'value': 4}}}}}) as result:
        assert result.success
    with pytest.raises(DagsterInvariantViolationError, match='run that is not failed'):
        ReexecutionOptions.from_failure(result.run_id, instance)

def highly_nested_job():
    if False:
        while True:
            i = 10

    @op
    def emit_one():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @op
    def add_one(x):
        if False:
            while True:
                i = 10
        return x + 1

    @job(executor_def=in_process_executor)
    def the_job():
        if False:
            i = 10
            return i + 15
        add_one.alias('add_one_outer')(add_one.alias('add_one_middle')(add_one.alias('add_one_inner')(emit_one())))
    return the_job

def test_reexecution_selection_syntax(instance):
    if False:
        print('Hello World!')
    result = execute_job(reconstructable(highly_nested_job), instance)
    assert result.success
    options_upstream = ReexecutionOptions(parent_run_id=result.run_id, step_selection=['*add_one_middle'])
    result = execute_job(reconstructable(highly_nested_job), instance, reexecution_options=options_upstream)
    assert result.success
    assert len(result.get_step_success_events()) == 3
    options_downstream = ReexecutionOptions(parent_run_id=result.run_id, step_selection=['*add_one_middle'])
    result = execute_job(reconstructable(highly_nested_job), instance, reexecution_options=options_downstream)
    assert result.success
    assert len(result.get_step_success_events()) == 3
    options_upstream = ReexecutionOptions(parent_run_id=result.run_id, step_selection=['++add_one_outer'])
    result = execute_job(reconstructable(highly_nested_job), instance, reexecution_options=options_upstream)
    assert result.success
    assert len(result.get_step_success_events()) == 3
    options_downstream = ReexecutionOptions(parent_run_id=result.run_id, step_selection=['emit_one++'])
    result = execute_job(reconstructable(highly_nested_job), instance, reexecution_options=options_downstream)
    assert result.success
    assert len(result.get_step_success_events()) == 3
    options_overlap = ReexecutionOptions(parent_run_id=result.run_id, step_selection=['++add_one_outer', 'emit_one++'])
    result = execute_job(reconstructable(highly_nested_job), instance, reexecution_options=options_overlap)
    assert result.success
    assert len(result.get_step_success_events()) == 4

def get_asset_job():
    if False:
        return 10

    @asset
    def downstream_asset(upstream_asset):
        if False:
            return 10
        return upstream_asset

    @asset
    def upstream_asset():
        if False:
            print('Hello World!')
        return 5
    the_job = define_asset_job(name='the_job', selection=['downstream_asset', 'upstream_asset'])

    @repository
    def the_repo():
        if False:
            i = 10
            return i + 15
        return [the_job, downstream_asset, upstream_asset]
    job_def = the_repo.get_job('the_job')
    return job_def

def test_asset_selection():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        result = execute_job(reconstructable(get_asset_job), instance, asset_selection=[AssetKey('upstream_asset')])
        assert result.success
        assert len(result.get_step_success_events()) == 1
        assert result.get_step_success_events()[0].step_key == 'upstream_asset'

@op(out={'a': Out(is_required=False), 'b': Out(is_required=False)})
def a_or_b():
    if False:
        return 10
    yield Output('wow', 'a')

@op
def echo(x):
    if False:
        while True:
            i = 10
    return x

@op
def fail_once(context: OpExecutionContext, x):
    if False:
        i = 10
        return i + 15
    key = context.op_handle.name
    if context.instance.run_storage.get_cursor_values({key}).get(key):
        return x
    context.instance.run_storage.set_cursor_values({key: 'true'})
    raise Exception('failed (just this once)')

@job(executor_def=in_process_executor)
def branching_job():
    if False:
        return 10
    (a, b) = a_or_b()
    echo([fail_once.alias('fail_once_a')(a), fail_once.alias('fail_once_b')(b)])

def test_branching():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test() as instance:
        result = execute_job(reconstructable(branching_job), instance)
        assert not result.success
        result_2 = execute_job(reconstructable(branching_job), instance, reexecution_options=ReexecutionOptions.from_failure(result.run_id, instance))
        assert result_2.success
        success_steps = {ev.step_key for ev in result_2.get_step_success_events()}
        assert success_steps == {'fail_once_a', 'echo'}