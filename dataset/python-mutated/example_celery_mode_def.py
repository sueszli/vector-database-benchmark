from dagster import job
from dagster_celery_k8s.executor import celery_k8s_job_executor

@job(executor_def=celery_k8s_job_executor)
def celery_enabled_job():
    if False:
        i = 10
        return i + 15
    pass