from dagster_docker import docker_executor
from dagster import job

@job(executor_def=docker_executor)
def docker_job():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_mode():
    if False:
        return 10
    assert docker_job