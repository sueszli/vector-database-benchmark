from typing import Dict, TypeVar
from dagster import job, op
T = TypeVar('T')

def _set_key_value(ddict: Dict[str, object], key: str, value: T) -> T:
    if False:
        while True:
            i = 10
    ddict[key] = value
    return value

def test_execute_op_with_dep_only_inputs_no_api():
    if False:
        for i in range(10):
            print('nop')
    did_run_dict = {}

    @op
    def step_one_op(_):
        if False:
            return 10
        _set_key_value(did_run_dict, 'step_one', True)

    @op
    def step_two_op(_, _in):
        if False:
            for i in range(10):
                print('nop')
        _set_key_value(did_run_dict, 'step_two', True)

    @job
    def foo_job():
        if False:
            for i in range(10):
                print('nop')
        step_two_op(step_one_op())
    result = foo_job.execute_in_process()
    assert result.success
    assert did_run_dict['step_one'] is True
    assert did_run_dict['step_two'] is True

def test_execute_op_with_dep_only_inputs_with_api():
    if False:
        return 10
    did_run_dict = {}

    @op
    def step_one_op(_):
        if False:
            print('Hello World!')
        _set_key_value(did_run_dict, 'step_one', True)

    @op
    def step_two_op(_, _in):
        if False:
            i = 10
            return i + 15
        _set_key_value(did_run_dict, 'step_two', True)

    @job
    def foo_job():
        if False:
            for i in range(10):
                print('nop')
        step_two_op(step_one_op())
    assert foo_job.execute_in_process().success
    assert did_run_dict['step_one'] is True
    assert did_run_dict['step_two'] is True