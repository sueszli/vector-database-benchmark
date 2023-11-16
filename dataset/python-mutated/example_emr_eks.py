from __future__ import annotations
import json
import subprocess
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.eks import ClusterStates, NodegroupStates
from airflow.providers.amazon.aws.operators.eks import EksCreateClusterOperator, EksDeleteClusterOperator
from airflow.providers.amazon.aws.operators.emr import EmrContainerOperator, EmrEksCreateClusterOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3CreateObjectOperator, S3DeleteBucketOperator
from airflow.providers.amazon.aws.sensors.eks import EksClusterStateSensor, EksNodegroupStateSensor
from airflow.providers.amazon.aws.sensors.emr import EmrContainerSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_emr_eks'
ROLE_ARN_KEY = 'ROLE_ARN'
JOB_ROLE_ARN_KEY = 'JOB_ROLE_ARN'
JOB_ROLE_NAME_KEY = 'JOB_ROLE_NAME'
SUBNETS_KEY = 'SUBNETS'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).add_variable(JOB_ROLE_ARN_KEY).add_variable(JOB_ROLE_NAME_KEY).add_variable(SUBNETS_KEY, split_string=True).build()
S3_FILE_NAME = 'pi.py'
S3_FILE_CONTENT = '\nk = 1\ns = 0\n\nfor i in range(1000000):\n    if i % 2 == 0:\n        s += 4/k\n    else:\n        s -= 4/k\n\n    k += 2\n\nprint(s)\n'

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

@task
def enable_access_emr_on_eks(cluster, ns):
    if False:
        for i in range(10):
            print('nop')
    file = 'https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz'
    commands = f'\n        curl --silent --location "{file}" | tar xz -C /tmp &&\n        sudo mv /tmp/eksctl /usr/local/bin &&\n        eksctl create iamidentitymapping --cluster {cluster} --namespace {ns} --service-name "emr-containers"\n    '
    build = subprocess.Popen(commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (_, err) = build.communicate()
    if build.returncode != 0:
        raise RuntimeError(err)

@task
def create_iam_oidc_identity_provider(cluster_name):
    if False:
        return 10
    command = f'eksctl utils associate-iam-oidc-provider --cluster {cluster_name} --approve'
    build = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (_, err) = build.communicate()
    if build.returncode != 0:
        raise RuntimeError(err)

@task
def delete_iam_oidc_identity_provider(cluster_name):
    if False:
        print('Hello World!')
    oidc_provider_issuer_url = boto3.client('eks').describe_cluster(name=cluster_name)['cluster']['identity']['oidc']['issuer']
    oidc_provider_issuer_endpoint = oidc_provider_issuer_url.replace('https://', '')
    account_id = boto3.client('sts').get_caller_identity()['Account']
    boto3.client('iam').delete_open_id_connect_provider(OpenIDConnectProviderArn=f'arn:aws:iam::{account_id}:oidc-provider/{oidc_provider_issuer_endpoint}')

@task
def update_trust_policy_execution_role(cluster_name, cluster_namespace, role_name):
    if False:
        i = 10
        return i + 15
    client = boto3.client('iam')
    role_trust_policy = client.get_role(RoleName=role_name)['Role']['AssumeRolePolicyDocument']
    role_trust_policy['Statement'] = [statement for statement in role_trust_policy['Statement'] if statement['Action'] != 'sts:AssumeRoleWithWebIdentity']
    client.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(role_trust_policy))
    commands = f'aws emr-containers update-role-trust-policy --cluster-name {cluster_name} --namespace {cluster_namespace} --role-name {role_name}'
    build = subprocess.Popen(commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (_, err) = build.communicate()
    if build.returncode != 0:
        raise RuntimeError(err)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_virtual_cluster(virtual_cluster_id):
    if False:
        for i in range(10):
            print('nop')
    boto3.client('emr-containers').delete_virtual_cluster(id=virtual_cluster_id)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    role_arn = test_context[ROLE_ARN_KEY]
    subnets = test_context[SUBNETS_KEY]
    job_role_arn = test_context[JOB_ROLE_ARN_KEY]
    job_role_name = test_context[JOB_ROLE_NAME_KEY]
    s3_bucket_name = f'{env_id}-bucket'
    eks_cluster_name = f'{env_id}-cluster'
    virtual_cluster_name = f'{env_id}-virtual-cluster'
    nodegroup_name = f'{env_id}-nodegroup'
    eks_namespace = 'default'
    launch_template_name = f'{env_id}-launch-template'
    job_driver_arg = {'sparkSubmitJobDriver': {'entryPoint': f's3://{s3_bucket_name}/{S3_FILE_NAME}', 'sparkSubmitParameters': '--conf spark.executors.instances=2 --conf spark.executors.memory=2G --conf spark.executor.cores=2 --conf spark.driver.cores=1'}}
    configuration_overrides_arg = {'monitoringConfiguration': {'cloudWatchMonitoringConfiguration': {'logGroupName': '/emr-eks-jobs', 'logStreamNamePrefix': 'airflow'}}}
    create_bucket = S3CreateBucketOperator(task_id='create_bucket', bucket_name=s3_bucket_name)
    upload_s3_file = S3CreateObjectOperator(task_id='upload_s3_file', s3_bucket=s3_bucket_name, s3_key=S3_FILE_NAME, data=S3_FILE_CONTENT)
    create_cluster_and_nodegroup = EksCreateClusterOperator(task_id='create_cluster_and_nodegroup', cluster_name=eks_cluster_name, nodegroup_name=nodegroup_name, cluster_role_arn=role_arn, nodegroup_role_arn=role_arn, resources_vpc_config={'subnetIds': subnets}, create_nodegroup_kwargs={'launchTemplate': {'name': launch_template_name}})
    await_create_nodegroup = EksNodegroupStateSensor(task_id='await_create_nodegroup', cluster_name=eks_cluster_name, nodegroup_name=nodegroup_name, target_state=NodegroupStates.ACTIVE, poke_interval=10)
    create_emr_eks_cluster = EmrEksCreateClusterOperator(task_id='create_emr_eks_cluster', virtual_cluster_name=virtual_cluster_name, eks_cluster_name=eks_cluster_name, eks_namespace=eks_namespace)
    job_starter = EmrContainerOperator(task_id='start_job', virtual_cluster_id=str(create_emr_eks_cluster.output), execution_role_arn=job_role_arn, release_label='emr-6.3.0-latest', job_driver=job_driver_arg, configuration_overrides=configuration_overrides_arg, name='pi.py')
    job_starter.wait_for_completion = False
    job_waiter = EmrContainerSensor(task_id='job_waiter', virtual_cluster_id=str(create_emr_eks_cluster.output), job_id=str(job_starter.output))
    delete_eks_cluster = EksDeleteClusterOperator(task_id='delete_eks_cluster', cluster_name=eks_cluster_name, force_delete_compute=True, trigger_rule=TriggerRule.ALL_DONE)
    await_delete_eks_cluster = EksClusterStateSensor(task_id='await_delete_eks_cluster', cluster_name=eks_cluster_name, target_state=ClusterStates.NONEXISTENT, trigger_rule=TriggerRule.ALL_DONE, poke_interval=10)
    delete_bucket = S3DeleteBucketOperator(task_id='delete_bucket', bucket_name=s3_bucket_name, force_delete=True, trigger_rule=TriggerRule.ALL_DONE)
    chain(test_context, create_bucket, upload_s3_file, create_launch_template(launch_template_name), create_cluster_and_nodegroup, await_create_nodegroup, enable_access_emr_on_eks(eks_cluster_name, eks_namespace), create_iam_oidc_identity_provider(eks_cluster_name), update_trust_policy_execution_role(eks_cluster_name, eks_namespace, job_role_name), create_emr_eks_cluster, job_starter, job_waiter, delete_iam_oidc_identity_provider(eks_cluster_name), delete_virtual_cluster(str(create_emr_eks_cluster.output)), delete_eks_cluster, await_delete_eks_cluster, delete_launch_template(launch_template_name), delete_bucket)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)