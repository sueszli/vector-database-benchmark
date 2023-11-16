from typing import Any, Dict
import pytest
from dagster import DagsterInvalidConfigError, job, op

def test_dict_input():
    if False:
        while True:
            i = 10

    @op
    def the_op(x: Dict[str, str]):
        if False:
            return 10
        assert x == {'foo': 'bar'}

    @job
    def the_job():
        if False:
            for i in range(10):
                print('nop')
        the_op()
    assert the_job.execute_in_process(run_config={'ops': {'the_op': {'inputs': {'x': {'foo': 'bar'}}}}}).success

    @job
    def the_job_top_lvl_input(x):
        if False:
            print('Hello World!')
        the_op(x)
    assert the_job_top_lvl_input.execute_in_process(run_config={'inputs': {'x': {'foo': 'bar'}}}).success

def test_any_dict_input():
    if False:
        i = 10
        return i + 15

    @op
    def the_op(x: Dict[str, Any]):
        if False:
            while True:
                i = 10
        assert x == {'foo': 'bar'}

    @job
    def the_job():
        if False:
            for i in range(10):
                print('nop')
        the_op()
    assert the_job.execute_in_process(run_config={'ops': {'the_op': {'inputs': {'x': {'foo': {'value': 'bar'}}}}}}).success

    @job
    def the_job_top_lvl_input(x):
        if False:
            i = 10
            return i + 15
        the_op(x)
    assert the_job_top_lvl_input.execute_in_process(run_config={'inputs': {'x': {'foo': {'value': 'bar'}}}}).success

def test_malformed_input_schema_dict():
    if False:
        for i in range(10):
            print('nop')

    @op
    def the_op(_x: Dict[str, Any]):
        if False:
            print('Hello World!')
        pass

    @job
    def the_job(x):
        if False:
            while True:
                i = 10
        the_op(x)
    with pytest.raises(DagsterInvalidConfigError):
        the_job.execute_in_process(run_config={'inputs': {'x': {'foo': 'bar'}}})
    with pytest.raises(DagsterInvalidConfigError):
        the_job.execute_in_process(run_config={'inputs': {'x': {'foo': {'foo': 'bar'}}}})