from dagster import reconstructable
from dagster._core.definitions import op
from dagster._core.definitions.decorators.job_decorator import job
from dagster._core.execution.api import execute_job
from dagster._core.test_utils import instance_for_test

@op(tags={'dagster/priority': '-1'})
def low(_):
    if False:
        print('Hello World!')
    pass

@op
def none(_):
    if False:
        print('Hello World!')
    pass

@op(tags={'dagster/priority': '1'})
def high(_):
    if False:
        while True:
            i = 10
    pass

@job
def priority_test():
    if False:
        while True:
            i = 10
    none()
    low()
    high()
    none()
    low()
    high()

def test_priorities():
    if False:
        print('Hello World!')
    result = priority_test.execute_in_process()
    assert result.success
    assert [str(event.node_handle) for event in result.get_step_success_events()] == ['high', 'high_2', 'none', 'none_2', 'low', 'low_2']

def test_priorities_mp():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test() as instance:
        recon_job = reconstructable(priority_test)
        with execute_job(recon_job, run_config={'execution': {'config': {'multiprocess': {'max_concurrent': 1}}}}, instance=instance) as result:
            assert result.success
            assert [str(event.node_handle) for event in result.get_step_success_events()] == ['high', 'high_2', 'none', 'none_2', 'low', 'low_2']