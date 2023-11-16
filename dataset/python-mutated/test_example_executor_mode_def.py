from dagster_k8s import k8s_job_executor
from dagster import job

@job(executor_def=k8s_job_executor)
def k8s_job():
    if False:
        i = 10
        return i + 15
    pass

def test_mode():
    if False:
        while True:
            i = 10
    assert k8s_job