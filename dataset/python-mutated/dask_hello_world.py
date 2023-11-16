from dagster_dask import dask_executor
from dagster import job, op

@op
def hello_world():
    if False:
        return 10
    return 'Hello, World!'

@job(executor_def=dask_executor)
def local_dask_job():
    if False:
        return 10
    hello_world()