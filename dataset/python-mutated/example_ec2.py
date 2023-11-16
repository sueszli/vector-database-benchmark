from __future__ import annotations
from datetime import datetime
from operator import itemgetter
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.ec2 import EC2CreateInstanceOperator, EC2StartInstanceOperator, EC2StopInstanceOperator, EC2TerminateInstanceOperator
from airflow.providers.amazon.aws.sensors.ec2 import EC2InstanceStateSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_ec2'
sys_test_context_task = SystemTestContextBuilder().build()

@task
def get_latest_ami_id():
    if False:
        print('Hello World!')
    'Returns the AMI ID of the most recently-created Amazon Linux image'
    image_prefix = 'Amazon Linux*'
    images = boto3.client('ec2').describe_images(Filters=[{'Name': 'description', 'Values': [image_prefix]}, {'Name': 'architecture', 'Values': ['arm64']}], Owners=['amazon'])
    return max(images['Images'], key=itemgetter('CreationDate'))['ImageId']

@task
def create_key_pair(key_name: str):
    if False:
        i = 10
        return i + 15
    client = boto3.client('ec2')
    key_pair_id = client.create_key_pair(KeyName=key_name)['KeyName']
    client.get_waiter('key_pair_exists').wait(KeyNames=[key_pair_id])
    return key_pair_id

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_key_pair(key_pair_id: str):
    if False:
        for i in range(10):
            print('nop')
    boto3.client('ec2').delete_key_pair(KeyName=key_pair_id)

@task
def parse_response(instance_ids: list):
    if False:
        for i in range(10):
            print('nop')
    return instance_ids[0]
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    instance_name = f'{env_id}-instance'
    key_name = create_key_pair(key_name=f'{env_id}_key_pair')
    image_id = get_latest_ami_id()
    config = {'InstanceType': 't4g.micro', 'KeyName': key_name, 'TagSpecifications': [{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': instance_name}]}], 'MetadataOptions': {'HttpEndpoint': 'enabled', 'HttpTokens': 'required'}}
    create_instance = EC2CreateInstanceOperator(task_id='create_instance', image_id=image_id, max_count=1, min_count=1, config=config)
    create_instance.wait_for_completion = True
    instance_id = parse_response(create_instance.output)
    stop_instance = EC2StopInstanceOperator(task_id='stop_instance', instance_id=instance_id)
    stop_instance.trigger_rule = TriggerRule.ALL_DONE
    start_instance = EC2StartInstanceOperator(task_id='start_instance', instance_id=instance_id)
    await_instance = EC2InstanceStateSensor(task_id='await_instance', instance_id=instance_id, target_state='running')
    terminate_instance = EC2TerminateInstanceOperator(task_id='terminate_instance', instance_ids=instance_id, wait_for_completion=True)
    terminate_instance.trigger_rule = TriggerRule.ALL_DONE
    chain(test_context, key_name, image_id, create_instance, instance_id, stop_instance, start_instance, await_instance, terminate_instance, delete_key_pair(key_name))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)