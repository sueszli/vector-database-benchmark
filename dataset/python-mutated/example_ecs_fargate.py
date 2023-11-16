from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.ecs import EcsTaskStates
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.providers.amazon.aws.sensors.ecs import EcsTaskStateSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_ecs_fargate'
SUBNETS_KEY = 'SUBNETS'
SECURITY_GROUPS_KEY = 'SECURITY_GROUPS'
sys_test_context_task = SystemTestContextBuilder().add_variable(SUBNETS_KEY, split_string=True).add_variable(SECURITY_GROUPS_KEY, split_string=True).build()

@task
def create_cluster(cluster_name: str) -> None:
    if False:
        return 10
    'Creates an ECS cluster.'
    boto3.client('ecs').create_cluster(clusterName=cluster_name)

@task
def register_task_definition(task_name: str, container_name: str) -> str:
    if False:
        print('Hello World!')
    'Creates a Task Definition.'
    response = boto3.client('ecs').register_task_definition(family=task_name, cpu='256', memory='512', containerDefinitions=[{'name': container_name, 'image': 'ubuntu', 'workingDirectory': '/usr/bin', 'entryPoint': ['sh', '-c'], 'command': ['ls']}], requiresCompatibilities=['FARGATE'], networkMode='awsvpc')
    return response['taskDefinition']['taskDefinitionArn']

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_task_definition(task_definition_arn: str) -> None:
    if False:
        while True:
            i = 10
    'Deletes the Task Definition.'
    boto3.client('ecs').deregister_task_definition(taskDefinition=task_definition_arn)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_cluster(cluster_name: str) -> None:
    if False:
        return 10
    'Deletes the ECS cluster.'
    boto3.client('ecs').delete_cluster(cluster=cluster_name)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    cluster_name = f'{env_id}-test-cluster'
    container_name = f'{env_id}-test-container'
    task_definition_name = f'{env_id}-test-definition'
    create_task_definition = register_task_definition(task_definition_name, container_name)
    hello_world = EcsRunTaskOperator(task_id='hello_world', cluster=cluster_name, task_definition=task_definition_name, launch_type='FARGATE', overrides={'containerOverrides': [{'name': container_name, 'command': ['echo', 'hello', 'world']}]}, network_configuration={'awsvpcConfiguration': {'subnets': test_context[SUBNETS_KEY], 'securityGroups': test_context[SECURITY_GROUPS_KEY], 'assignPublicIp': 'ENABLED'}})
    hello_world.wait_for_completion = False
    await_task_finish = EcsTaskStateSensor(task_id='await_task_finish', cluster=cluster_name, task=hello_world.output['ecs_task_arn'], target_state=EcsTaskStates.STOPPED, failure_states={EcsTaskStates.NONE})
    chain(test_context, create_cluster(cluster_name), create_task_definition, hello_world, await_task_finish, delete_task_definition(create_task_definition), delete_cluster(cluster_name))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)