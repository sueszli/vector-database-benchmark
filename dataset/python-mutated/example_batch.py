from __future__ import annotations
import logging
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.batch import BatchCreateComputeEnvironmentOperator, BatchOperator
from airflow.providers.amazon.aws.sensors.batch import BatchComputeEnvironmentSensor, BatchJobQueueSensor, BatchSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder, prune_logs, split_string
log = logging.getLogger(__name__)
DAG_ID = 'example_batch'
ROLE_ARN_KEY = 'ROLE_ARN'
SUBNETS_KEY = 'SUBNETS'
SECURITY_GROUPS_KEY = 'SECURITY_GROUPS'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).add_variable(SUBNETS_KEY).add_variable(SECURITY_GROUPS_KEY).build()
JOB_OVERRIDES: dict = {}

@task
def create_job_definition(role_arn, job_definition_name):
    if False:
        for i in range(10):
            print('nop')
    boto3.client('batch').register_job_definition(type='container', containerProperties={'command': ['sleep', '2'], 'executionRoleArn': role_arn, 'image': 'busybox', 'resourceRequirements': [{'value': '1', 'type': 'VCPU'}, {'value': '2048', 'type': 'MEMORY'}], 'networkConfiguration': {'assignPublicIp': 'ENABLED'}}, jobDefinitionName=job_definition_name, platformCapabilities=['FARGATE'])

@task
def create_job_queue(job_compute_environment_name, job_queue_name):
    if False:
        for i in range(10):
            print('nop')
    boto3.client('batch').create_job_queue(computeEnvironmentOrder=[{'computeEnvironment': job_compute_environment_name, 'order': 1}], jobQueueName=job_queue_name, priority=1, state='ENABLED')

@task(trigger_rule=TriggerRule.ONE_FAILED)
def describe_job(job_id):
    if False:
        while True:
            i = 10
    client = boto3.client('batch')
    response = client.describe_jobs(jobs=[job_id])
    log.info('Describing the job %s for debugging purposes', job_id)
    log.info(response['jobs'])

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_job_definition(job_definition_name):
    if False:
        i = 10
        return i + 15
    client = boto3.client('batch')
    response = client.describe_job_definitions(jobDefinitionName=job_definition_name, status='ACTIVE')
    for job_definition in response['jobDefinitions']:
        client.deregister_job_definition(jobDefinition=job_definition['jobDefinitionArn'])

@task(trigger_rule=TriggerRule.ALL_DONE)
def disable_compute_environment(job_compute_environment_name):
    if False:
        i = 10
        return i + 15
    boto3.client('batch').update_compute_environment(computeEnvironment=job_compute_environment_name, state='DISABLED')

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_compute_environment(job_compute_environment_name):
    if False:
        i = 10
        return i + 15
    boto3.client('batch').delete_compute_environment(computeEnvironment=job_compute_environment_name)

@task(trigger_rule=TriggerRule.ALL_DONE)
def disable_job_queue(job_queue_name):
    if False:
        i = 10
        return i + 15
    boto3.client('batch').update_job_queue(jobQueue=job_queue_name, state='DISABLED')

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_job_queue(job_queue_name):
    if False:
        print('Hello World!')
    boto3.client('batch').delete_job_queue(jobQueue=job_queue_name)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    batch_job_name: str = f'{env_id}-test-job'
    batch_job_definition_name: str = f'{env_id}-test-job-definition'
    batch_job_compute_environment_name: str = f'{env_id}-test-job-compute-environment'
    batch_job_queue_name: str = f'{env_id}-test-job-queue'
    security_groups = split_string(test_context[SECURITY_GROUPS_KEY])
    subnets = split_string(test_context[SUBNETS_KEY])
    create_compute_environment = BatchCreateComputeEnvironmentOperator(task_id='create_compute_environment', compute_environment_name=batch_job_compute_environment_name, environment_type='MANAGED', state='ENABLED', compute_resources={'type': 'FARGATE', 'maxvCpus': 10, 'securityGroupIds': security_groups, 'subnets': subnets})
    wait_for_compute_environment_valid = BatchComputeEnvironmentSensor(task_id='wait_for_compute_environment_valid', compute_environment=batch_job_compute_environment_name)
    wait_for_compute_environment_valid.poke_interval = 1
    wait_for_job_queue_valid = BatchJobQueueSensor(task_id='wait_for_job_queue_valid', job_queue=batch_job_queue_name)
    wait_for_job_queue_valid.poke_interval = 1
    submit_batch_job = BatchOperator(task_id='submit_batch_job', job_name=batch_job_name, job_queue=batch_job_queue_name, job_definition=batch_job_definition_name, overrides=JOB_OVERRIDES)
    submit_batch_job.wait_for_completion = False
    wait_for_batch_job = BatchSensor(task_id='wait_for_batch_job', job_id=submit_batch_job.output)
    wait_for_batch_job.poke_interval = 10
    wait_for_compute_environment_disabled = BatchComputeEnvironmentSensor(task_id='wait_for_compute_environment_disabled', compute_environment=batch_job_compute_environment_name, poke_interval=1)
    wait_for_job_queue_modified = BatchJobQueueSensor(task_id='wait_for_job_queue_modified', job_queue=batch_job_queue_name, poke_interval=1)
    wait_for_job_queue_deleted = BatchJobQueueSensor(task_id='wait_for_job_queue_deleted', job_queue=batch_job_queue_name, treat_non_existing_as_deleted=True, poke_interval=10)
    log_cleanup = prune_logs([('/aws/batch/job', env_id)])
    chain(test_context, security_groups, subnets, create_job_definition(test_context[ROLE_ARN_KEY], batch_job_definition_name), create_compute_environment, wait_for_compute_environment_valid, create_job_queue(batch_job_compute_environment_name, batch_job_queue_name), wait_for_job_queue_valid, submit_batch_job, wait_for_batch_job, describe_job(submit_batch_job.output), disable_job_queue(batch_job_queue_name), wait_for_job_queue_modified, delete_job_queue(batch_job_queue_name), wait_for_job_queue_deleted, disable_compute_environment(batch_job_compute_environment_name), wait_for_compute_environment_disabled, delete_compute_environment(batch_job_compute_environment_name), delete_job_definition(batch_job_definition_name), log_cleanup)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)