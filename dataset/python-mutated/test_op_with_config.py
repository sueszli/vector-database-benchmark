import pytest
from dagster import ConfigMapping, DagsterInvalidConfigError, Field, In, String, graph, job, op
from dagster._core.storage.input_manager import input_manager

def test_basic_op_with_config():
    if False:
        return 10
    did_get = {}

    @op(name='op_with_context', ins={}, out={}, config_schema={'some_config': Field(String)})
    def op_with_context(context):
        if False:
            print('Hello World!')
        did_get['yep'] = context.op_config

    @job
    def job_def():
        if False:
            while True:
                i = 10
        op_with_context()
    job_def.execute_in_process({'ops': {'op_with_context': {'config': {'some_config': 'foo'}}}})
    assert 'yep' in did_get
    assert 'some_config' in did_get['yep']

def test_config_arg_mismatch():
    if False:
        while True:
            i = 10

    def _t_fn(*_args):
        if False:
            i = 10
            return i + 15
        raise Exception('should not reach')

    @op(name='op_with_context', ins={}, out={}, config_schema={'some_config': Field(String)})
    def op_with_context(context):
        if False:
            i = 10
            return i + 15
        raise Exception('should not reach')

    @job
    def job_def():
        if False:
            print('Hello World!')
        op_with_context()
    with pytest.raises(DagsterInvalidConfigError):
        job_def.execute_in_process({'ops': {'op_with_context': {'config': {'some_config': 1}}}})

def test_op_not_found():
    if False:
        i = 10
        return i + 15

    @op(name='find_me_op', ins={}, out={})
    def find_me_op(_):
        if False:
            print('Hello World!')
        raise Exception('should not reach')

    @job
    def job_def():
        if False:
            for i in range(10):
                print('nop')
        find_me_op()
    with pytest.raises(DagsterInvalidConfigError):
        job_def.execute_in_process({'ops': {'not_found': {'config': {'some_config': 1}}}})

def test_extra_config_ignored_default_input():
    if False:
        print('Hello World!')

    @op(config_schema={'some_config': str})
    def op1(_):
        if False:
            for i in range(10):
                print('nop')
        return 'public.table_1'

    @op
    def op2(_, input_table='public.table_1'):
        if False:
            for i in range(10):
                print('nop')
        return input_table

    @job
    def my_job():
        if False:
            print('Hello World!')
        op2(op1())
    run_config = {'ops': {'op1': {'config': {'some_config': 'a'}}}}
    assert my_job.execute_in_process(run_config=run_config).success
    assert my_job.execute_in_process(run_config=run_config, op_selection=['op2']).success
    with pytest.raises(DagsterInvalidConfigError):
        my_job.execute_in_process({'ops': {'solid_1': {'config': {'some_config': 'a'}}}}, op_selection=['op2'])

def test_extra_config_ignored_no_default_input():
    if False:
        return 10

    @op(config_schema={'some_config': str})
    def op1(_):
        if False:
            return 10
        return 'public.table_1'

    @op
    def op2(_, input_table):
        if False:
            print('Hello World!')
        return input_table

    @job
    def my_job():
        if False:
            print('Hello World!')
        op2(op1())
    run_config = {'ops': {'op1': {'config': {'some_config': 'a'}}}}
    assert my_job.execute_in_process(run_config=run_config).success
    with pytest.raises(DagsterInvalidConfigError):
        my_job.execute_in_process(run_config=run_config, op_selection=['op2'])
    run_config['ops']['op2'] = {'inputs': {'input_table': {'value': 'public.table_1'}}}
    assert my_job.execute_in_process(run_config=run_config, op_selection=['op2']).success
    assert my_job.execute_in_process(run_config=run_config, op_selection=['op1']).success

def test_extra_config_ignored_composites():
    if False:
        i = 10
        return i + 15

    @op(config_schema={'some_config': str})
    def op1(_):
        if False:
            return 10
        return 'public.table_1'

    @graph(config=ConfigMapping(config_schema={'wrapped_config': str}, config_fn=lambda cfg: {'op1': {'config': {'some_config': cfg['wrapped_config']}}}))
    def graph1():
        if False:
            return 10
        return op1()

    @op
    def op2(_, input_table='public.table'):
        if False:
            return 10
        return input_table

    @graph
    def graph2(input_table):
        if False:
            return 10
        return op2(input_table)

    @job
    def my_job():
        if False:
            while True:
                i = 10
        graph2(graph1())
    run_config = {'ops': {'graph1': {'config': {'wrapped_config': 'a'}}}}
    assert my_job.execute_in_process(run_config=run_config).success
    assert my_job.execute_in_process(run_config=run_config, op_selection=['graph2']).success
    assert my_job.execute_in_process(run_config=run_config, op_selection=['graph1']).success

def test_extra_config_input_bug():
    if False:
        print('Hello World!')

    @op
    def root(_):
        if False:
            return 10
        return 'public.table_1'

    @op(config_schema={'some_config': str})
    def takes_input(_, input_table):
        if False:
            print('Hello World!')
        return input_table

    @job
    def my_job():
        if False:
            print('Hello World!')
        takes_input(root())
    run_config = {'ops': {'takes_input': {'config': {'some_config': 'a'}}}}
    assert my_job.execute_in_process(run_config=run_config).success
    assert my_job.execute_in_process(run_config=run_config, op_selection=['root']).success
    assert my_job.execute_in_process(op_selection=['root']).success

def test_extra_config_unsatisfied_input():
    if False:
        i = 10
        return i + 15

    @op
    def start(_, x):
        if False:
            while True:
                i = 10
        return x

    @op
    def end(_, x=1):
        if False:
            return 10
        return x

    @job
    def testing():
        if False:
            i = 10
            return i + 15
        end(start())
    assert testing.execute_in_process(run_config={'ops': {'start': {'inputs': {'x': {'value': 4}}}}}).success
    assert testing.execute_in_process(run_config={'ops': {'start': {'inputs': {'x': {'value': 4}}}}}, op_selection=['end']).success

def test_extra_config_unsatisfied_input_io_man():
    if False:
        i = 10
        return i + 15

    @input_manager(input_config_schema=int)
    def config_io_man(context):
        if False:
            while True:
                i = 10
        return context.config

    @op(ins={'x': In(input_manager_key='my_loader')})
    def start(_, x):
        if False:
            return 10
        return x

    @op
    def end(_, x=1):
        if False:
            while True:
                i = 10
        return x

    @job(resource_defs={'my_loader': config_io_man})
    def testing_io():
        if False:
            return 10
        end(start())
    assert testing_io.execute_in_process(run_config={'ops': {'start': {'inputs': {'x': 3}}}}).success
    assert testing_io.execute_in_process(run_config={'ops': {'start': {'inputs': {'x': 3}}}}, op_selection=['end']).success

def test_config_with_no_schema():
    if False:
        i = 10
        return i + 15

    @op
    def my_op(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.op_config == 5

    @job
    def my_job():
        if False:
            while True:
                i = 10
        my_op()
    my_job.execute_in_process(run_config={'ops': {'my_op': {'config': 5}}})