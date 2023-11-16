from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.eks import ClusterStates, NodegroupStates
from airflow.providers.amazon.aws.operators.eks import EksCreateClusterOperator, EksDeleteClusterOperator, EksPodOperator
from airflow.providers.amazon.aws.sensors.eks import EksClusterStateSensor, EksNodegroupStateSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
from tests.system.providers.amazon.aws.utils.k8s import get_describe_pod_operator
DAG_ID = 'example_eks_with_nodegroup_in_one_step'
ROLE_ARN_KEY = 'ROLE_ARN'
SUBNETS_KEY = 'SUBNETS'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).add_variable(SUBNETS_KEY, split_string=True).build()

@task
def create_launch_template(template_name: str):
    if False:
        for i in range(10):
            print('nop')
    boto3.client('ec2').create_launch_template(LaunchTemplateName=template_name, LaunchTemplateData={'MetadataOptions': {'HttpEndpoint': 'enabled', 'HttpTokens': 'required'}})

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_launch_template(template_name: str):
    if False:
        i = 10
        return i + 15
    boto3.client('ec2').delete_launch_template(LaunchTemplateName=template_name)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    cluster_name = f'{env_id}-cluster'
    nodegroup_name = f'{env_id}-nodegroup'
    launch_template_name = f'{env_id}-launch-template'
    create_cluster_and_nodegroup = EksCreateClusterOperator(task_id='create_cluster_and_nodegroup', cluster_name=cluster_name, nodegroup_name=nodegroup_name, cluster_role_arn=test_context[ROLE_ARN_KEY], nodegroup_role_arn=test_context[ROLE_ARN_KEY], resources_vpc_config={'subnetIds': test_context[SUBNETS_KEY]}, compute='nodegroup', create_nodegroup_kwargs={'launchTemplate': {'name': launch_template_name}})
    await_create_nodegroup = EksNodegroupStateSensor(task_id='await_create_nodegroup', cluster_name=cluster_name, nodegroup_name=nodegroup_name, target_state=NodegroupStates.ACTIVE, poke_interval=10)
    start_pod = EksPodOperator(task_id='start_pod', pod_name='test_pod', cluster_name=cluster_name, image='amazon/aws-cli:latest', cmds=['sh', '-c', 'echo Test Airflow; date'], labels={'demo': 'hello_world'}, get_logs=True, on_finish_action='keep_pod')
    describe_pod = get_describe_pod_operator(cluster_name, pod_name="{{ ti.xcom_pull(key='pod_name', task_ids='run_pod') }}")
    describe_pod.trigger_rule = TriggerRule.ONE_FAILED
    delete_nodegroup_and_cluster = EksDeleteClusterOperator(task_id='delete_nodegroup_and_cluster', cluster_name=cluster_name, force_delete_compute=True)
    delete_nodegroup_and_cluster.trigger_rule = TriggerRule.ALL_DONE
    await_delete_cluster = EksClusterStateSensor(task_id='await_delete_cluster', trigger_rule=TriggerRule.ALL_DONE, cluster_name=cluster_name, target_state=ClusterStates.NONEXISTENT, poke_interval=10)
    chain(test_context, create_launch_template(launch_template_name), create_cluster_and_nodegroup, await_create_nodegroup, start_pod, describe_pod, delete_nodegroup_and_cluster, await_delete_cluster, delete_launch_template(launch_template_name))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)