from dagster_k8s import k8s_job_executor
from dagster import OpExecutionContext, job, op

@op(tags={'dagster-k8s/config': {'container_config': {'resources': {'requests': {'cpu': '200m', 'memory': '32Mi'}}}}})
def my_op(context: OpExecutionContext):
    if False:
        print('Hello World!')
    context.log.info('running')

@job(executor_def=k8s_job_executor)
def my_job():
    if False:
        print('Hello World!')
    my_op()