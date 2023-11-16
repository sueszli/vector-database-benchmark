from __future__ import annotations
from airflow.providers.amazon.aws.hooks.glue_databrew import GlueDataBrewHook
from airflow.providers.amazon.aws.triggers.base import AwsBaseWaiterTrigger

class GlueDataBrewJobCompleteTrigger(AwsBaseWaiterTrigger):
    """
    Watches for a Glue DataBrew job, triggers when it finishes.

    :param job_name: Glue DataBrew job name
    :param run_id: the ID of the specific run to watch for that job
    :param delay: Number of seconds to wait between two checks. Default is 10 seconds.
    :param max_attempts: Maximum number of attempts to wait for the job to complete. Default is 60 attempts.
    :param aws_conn_id: The Airflow connection used for AWS credentials.
    """

    def __init__(self, job_name: str, run_id: str, aws_conn_id: str, delay: int=10, max_attempts: int=60, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(serialized_fields={'job_name': job_name, 'run_id': run_id}, waiter_name='job_complete', waiter_args={'Name': job_name, 'RunId': run_id}, failure_message=f'Error while waiting for job {job_name} with run id {run_id} to complete', status_message=f'Run id: {run_id}', status_queries=['State'], return_value=run_id, return_key='run_id', waiter_delay=delay, waiter_max_attempts=max_attempts, aws_conn_id=aws_conn_id)

    def hook(self) -> GlueDataBrewHook:
        if False:
            while True:
                i = 10
        return GlueDataBrewHook(aws_conn_id=self.aws_conn_id)