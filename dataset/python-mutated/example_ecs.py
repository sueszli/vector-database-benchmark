from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.ecs import EcsClusterStates
from airflow.providers.amazon.aws.operators.ecs import EcsCreateClusterOperator, EcsDeleteClusterOperator, EcsDeregisterTaskDefinitionOperator, EcsRegisterTaskDefinitionOperator, EcsRunTaskOperator
from airflow.providers.amazon.aws.sensors.ecs import EcsClusterStateSensor, EcsTaskDefinitionStateSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_ecs'
EXISTING_CLUSTER_NAME_KEY = 'CLUSTER_NAME'
EXISTING_CLUSTER_SUBNETS_KEY = 'SUBNETS'
sys_test_context_task = SystemTestContextBuilder().add_variable(EXISTING_CLUSTER_NAME_KEY).add_variable(EXISTING_CLUSTER_SUBNETS_KEY, split_string=True).build()

@task
def get_region():
    if False:
        while True:
            i = 10
    return boto3.session.Session().region_name

@task(trigger_rule=TriggerRule.ALL_DONE)
def clean_logs(group_name: str):
    if False:
        print('Hello World!')
    client = boto3.client('logs')
    client.delete_log_group(logGroupName=group_name)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    existing_cluster_name = test_context[EXISTING_CLUSTER_NAME_KEY]
    existing_cluster_subnets = test_context[EXISTING_CLUSTER_SUBNETS_KEY]
    new_cluster_name = f'{env_id}-cluster'
    container_name = f'{env_id}-container'
    family_name = f'{env_id}-task-definition'
    asg_name = f'{env_id}-asg'
    aws_region = get_region()
    log_group_name = f'/ecs_test/{env_id}'
    create_cluster = EcsCreateClusterOperator(task_id='create_cluster', cluster_name=new_cluster_name)
    create_cluster.wait_for_completion = False
    await_cluster = EcsClusterStateSensor(task_id='await_cluster', cluster_name=new_cluster_name)
    register_task = EcsRegisterTaskDefinitionOperator(task_id='register_task', family=family_name, container_definitions=[{'name': container_name, 'image': 'ubuntu', 'workingDirectory': '/usr/bin', 'entryPoint': ['sh', '-c'], 'command': ['ls'], 'logConfiguration': {'logDriver': 'awslogs', 'options': {'awslogs-group': log_group_name, 'awslogs-region': aws_region, 'awslogs-create-group': 'true', 'awslogs-stream-prefix': 'ecs'}}}], register_task_kwargs={'cpu': '256', 'memory': '512', 'networkMode': 'awsvpc'})
    await_task_definition = EcsTaskDefinitionStateSensor(task_id='await_task_definition', task_definition=register_task.output)
    run_task = EcsRunTaskOperator(task_id='run_task', cluster=existing_cluster_name, task_definition=register_task.output, overrides={'containerOverrides': [{'name': container_name, 'command': ['echo hello world']}]}, network_configuration={'awsvpcConfiguration': {'subnets': existing_cluster_subnets}}, awslogs_group=log_group_name, awslogs_region=aws_region, awslogs_stream_prefix=f'ecs/{container_name}')
    deregister_task = EcsDeregisterTaskDefinitionOperator(task_id='deregister_task', task_definition=register_task.output)
    deregister_task.trigger_rule = TriggerRule.ALL_DONE
    delete_cluster = EcsDeleteClusterOperator(task_id='delete_cluster', cluster_name=new_cluster_name)
    delete_cluster.trigger_rule = TriggerRule.ALL_DONE
    delete_cluster.wait_for_completion = False
    await_delete_cluster = EcsClusterStateSensor(task_id='await_delete_cluster', cluster_name=new_cluster_name, target_state=EcsClusterStates.INACTIVE)
    chain(test_context, aws_region, create_cluster, await_cluster, register_task, await_task_definition, run_task, deregister_task, delete_cluster, await_delete_cluster, clean_logs(log_group_name))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)