from __future__ import annotations
import json
from datetime import datetime
from typing import Any
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.ssm import SsmHook
from airflow.providers.amazon.aws.operators.emr import EmrAddStepsOperator, EmrCreateJobFlowOperator, EmrModifyClusterOperator, EmrTerminateJobFlowOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3DeleteBucketOperator
from airflow.providers.amazon.aws.sensors.emr import EmrJobFlowSensor, EmrStepSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_emr'
CONFIG_NAME = 'EMR Runtime Role Security Configuration'
EXECUTION_ROLE_ARN_KEY = 'EXECUTION_ROLE_ARN'
SECURITY_CONFIGURATION = {'AuthorizationConfiguration': {'IAMConfiguration': {'EnableApplicationScopedIAMRole': True}}, 'InstanceMetadataServiceConfiguration': {'MinimumInstanceMetadataServiceVersion': 2, 'HttpPutResponseHopLimit': 2}}
SPARK_STEPS = [{'Name': 'calculate_pi', 'ActionOnFailure': 'CONTINUE', 'HadoopJarStep': {'Jar': 'command-runner.jar', 'Args': ['/usr/lib/spark/bin/run-example', 'SparkPi', '10']}}]
JOB_FLOW_OVERRIDES: dict[str, Any] = {'Name': 'PiCalc', 'ReleaseLabel': 'emr-6.7.0', 'Applications': [{'Name': 'Spark'}], 'Instances': {'InstanceGroups': [{'Name': 'Primary node', 'Market': 'ON_DEMAND', 'InstanceRole': 'MASTER', 'InstanceType': 'm5.xlarge', 'InstanceCount': 1}], 'KeepJobFlowAliveWhenNoSteps': False, 'TerminationProtected': False}, 'Steps': SPARK_STEPS, 'JobFlowRole': 'EMR_EC2_DefaultRole', 'ServiceRole': 'EMR_DefaultRole'}

@task
def get_ami_id():
    if False:
        return 10
    '\n    Returns an AL2 AMI compatible with EMR\n    '
    return SsmHook(aws_conn_id=None).get_parameter_value('/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-ebs')

@task
def configure_security_config(config_name: str):
    if False:
        for i in range(10):
            print('nop')
    boto3.client('emr').create_security_configuration(Name=config_name, SecurityConfiguration=json.dumps(SECURITY_CONFIGURATION))

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_security_config(config_name: str):
    if False:
        for i in range(10):
            print('nop')
    boto3.client('emr').delete_security_configuration(Name=config_name)

@task
def get_step_id(step_ids: list):
    if False:
        for i in range(10):
            print('nop')
    return step_ids[0]
sys_test_context_task = SystemTestContextBuilder().add_variable(EXECUTION_ROLE_ARN_KEY).build()
with DAG(dag_id=DAG_ID, start_date=datetime(2021, 1, 1), schedule='@once', catchup=False, tags=['example']) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    config_name = f'{CONFIG_NAME}-{env_id}'
    execution_role_arn = test_context[EXECUTION_ROLE_ARN_KEY]
    s3_bucket = f'{env_id}-emr-bucket'
    JOB_FLOW_OVERRIDES['LogUri'] = f's3://{s3_bucket}/'
    JOB_FLOW_OVERRIDES['SecurityConfiguration'] = config_name
    JOB_FLOW_OVERRIDES['Instances']['InstanceGroups'][0]['CustomAmiId'] = get_ami_id()
    create_s3_bucket = S3CreateBucketOperator(task_id='create_s3_bucket', bucket_name=s3_bucket)
    create_security_configuration = configure_security_config(config_name)
    create_job_flow = EmrCreateJobFlowOperator(task_id='create_job_flow', job_flow_overrides=JOB_FLOW_OVERRIDES)
    modify_cluster = EmrModifyClusterOperator(task_id='modify_cluster', cluster_id=create_job_flow.output, step_concurrency_level=1)
    add_steps = EmrAddStepsOperator(task_id='add_steps', job_flow_id=create_job_flow.output, steps=SPARK_STEPS, execution_role_arn=execution_role_arn)
    add_steps.wait_for_completion = True
    add_steps.waiter_max_attempts = 90
    wait_for_step = EmrStepSensor(task_id='wait_for_step', job_flow_id=create_job_flow.output, step_id=get_step_id(add_steps.output))
    remove_cluster = EmrTerminateJobFlowOperator(task_id='remove_cluster', job_flow_id=create_job_flow.output)
    remove_cluster.trigger_rule = TriggerRule.ALL_DONE
    check_job_flow = EmrJobFlowSensor(task_id='check_job_flow', job_flow_id=create_job_flow.output)
    check_job_flow.poke_interval = 10
    delete_security_configuration = delete_security_config(config_name)
    delete_s3_bucket = S3DeleteBucketOperator(task_id='delete_s3_bucket', bucket_name=s3_bucket, force_delete=True, trigger_rule=TriggerRule.ALL_DONE)
    chain(test_context, create_s3_bucket, create_security_configuration, create_job_flow, modify_cluster, add_steps, wait_for_step, remove_cluster, check_job_flow, delete_security_configuration, delete_s3_bucket)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)