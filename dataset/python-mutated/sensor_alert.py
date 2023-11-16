import os
from dagster import run_failure_sensor, RunFailureSensorContext
from slack_sdk import WebClient

@run_failure_sensor
def my_slack_on_run_failure(context: RunFailureSensorContext):
    if False:
        for i in range(10):
            print('nop')
    slack_client = WebClient(token=os.environ['SLACK_DAGSTER_ETL_BOT_TOKEN'])
    slack_client.chat_postMessage(channel='#alert-channel', text=f'Job "{context.dagster_run.job_name}" failed. Error: {context.failure_event.message}')

def email_alert(_):
    if False:
        print('Hello World!')
    pass

@run_failure_sensor
def my_email_failure_sensor(context: RunFailureSensorContext):
    if False:
        i = 10
        return i + 15
    message = f'Job "{context.dagster_run.job_name}" failed. Error: {context.failure_event.message}'
    email_alert(message)
from dagster import op, job

@op
def fails():
    if False:
        return 10
    raise Exception('failure!')

@job
def my_job_fails():
    if False:
        return 10
    fails()
from dagster import DagsterInstance, build_run_status_sensor_context
instance = DagsterInstance.ephemeral()
result = my_job_fails.execute_in_process(instance=instance, raise_on_error=False)
dagster_run = result.dagster_run
dagster_event = result.get_job_failure_event()
run_failure_sensor_context = build_run_status_sensor_context(sensor_name='my_email_failure_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event).for_run_failure()
my_email_failure_sensor(run_failure_sensor_context)
from dagster_slack import make_slack_on_run_failure_sensor
slack_on_run_failure = make_slack_on_run_failure_sensor('#my_channel', os.getenv('MY_SLACK_TOKEN'))
from dagster import make_email_on_run_failure_sensor
email_on_run_failure = make_email_on_run_failure_sensor(email_from='no-reply@example.com', email_password=os.getenv('ALERT_EMAIL_PASSWORD'), email_to=['xxx@example.com', 'xyz@example.com'])
from dagster import run_status_sensor, RunStatusSensorContext, DagsterRunStatus

@run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
def my_slack_on_run_success(context: RunStatusSensorContext):
    if False:
        while True:
            i = 10
    slack_client = WebClient(token=os.environ['SLACK_DAGSTER_ETL_BOT_TOKEN'])
    slack_client.chat_postMessage(channel='#alert-channel', text=f'Job "{context.dagster_run.job_name}" succeeded.')

@run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
def my_email_sensor(context: RunStatusSensorContext):
    if False:
        for i in range(10):
            print('nop')
    message = f'Job "{context.dagster_run.job_name}" succeeded.'
    email_alert(message)

@op
def succeeds():
    if False:
        for i in range(10):
            print('nop')
    return 1

@job
def my_job_succeeds():
    if False:
        i = 10
        return i + 15
    succeeds()
instance = DagsterInstance.ephemeral()
result = my_job_succeeds.execute_in_process(instance=instance)
dagster_run = result.dagster_run
dagster_event = result.get_job_success_event()
run_status_sensor_context = build_run_status_sensor_context(sensor_name='my_email_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event)
my_email_sensor(run_status_sensor_context)
from dagster import SensorDefinition
from typing import List
my_jobs: List[SensorDefinition] = []

@job
def my_sensor_job():
    if False:
        while True:
            i = 10
    succeeds()
from dagster import Definitions
defs = Definitions(jobs=[my_sensor_job], sensors=[my_slack_on_run_success])