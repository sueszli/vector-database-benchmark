import sys
from dagster import job
from dagster._utils import file_relative_path
from dagster_dask import dask_executor
sys.path.append(file_relative_path(__file__, '../../../dagster-test/dagster_test/toys'))
from hammer import hammer

@job(executor_def=dask_executor)
def hammer_job():
    if False:
        for i in range(10):
            print('nop')
    hammer()