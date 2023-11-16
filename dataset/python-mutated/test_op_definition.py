import pytest
from dagster import DagsterInvariantViolationError, In, Nothing, OpDefinition, Out, Output, job, op

def test_op_def_direct():
    if False:
        while True:
            i = 10

    def the_op_fn(_, inputs):
        if False:
            return 10
        assert inputs['x'] == 5
        yield Output(inputs['x'] + 1, output_name='the_output')
    op_def = OpDefinition(the_op_fn, 'the_op', ins={'x': In(dagster_type=int)}, outs={'the_output': Out(int)})

    @job
    def the_job(x):
        if False:
            print('Hello World!')
        op_def(x)
    result = the_job.execute_in_process(input_values={'x': 5})
    assert result.success

def test_multi_out_implicit_none():
    if False:
        i = 10
        return i + 15

    @op(out={'a': Out(Nothing), 'b': Out(Nothing)})
    def implicit():
        if False:
            while True:
                i = 10
        pass
    implicit()

    @job
    def implicit_job():
        if False:
            return 10
        implicit()
    result = implicit_job.execute_in_process()
    assert result.success

    @op(out={'a': Out(Nothing), 'b': Out(Nothing, is_required=False)})
    def optional():
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(DagsterInvariantViolationError, match='has multiple outputs, but only one output was returned'):
        optional()

    @job
    def optional_job():
        if False:
            return 10
        optional()
    with pytest.raises(DagsterInvariantViolationError, match='has multiple outputs, but only one output was returned'):
        optional_job.execute_in_process()

    @op(out={'a': Out(), 'b': Out()})
    def untyped():
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvariantViolationError, match='has multiple outputs, but only one output was returned'):
        untyped()

    @job
    def untyped_job():
        if False:
            for i in range(10):
                print('nop')
        untyped()
    with pytest.raises(DagsterInvariantViolationError, match='has multiple outputs, but only one output was returned'):
        untyped_job.execute_in_process()