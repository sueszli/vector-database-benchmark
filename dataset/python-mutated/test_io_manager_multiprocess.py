from dagster import execute_job, job, multiprocess_executor, op, reconstructable
from dagster._core.test_utils import instance_for_test

@op
def op_a(_context):
    if False:
        for i in range(10):
            print('nop')
    return [1, 2, 3]

@op
def op_b(_context, _df):
    if False:
        print('Hello World!')
    return 1

@job(executor_def=multiprocess_executor)
def my_job():
    if False:
        i = 10
        return i + 15
    op_b(op_a())

def test_io_manager_with_multi_process_executor():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        with execute_job(reconstructable(my_job), instance=instance) as result:
            assert result.success
            assert result.output_for_node('op_b') == 1
            assert result.output_for_node('op_a') == [1, 2, 3]