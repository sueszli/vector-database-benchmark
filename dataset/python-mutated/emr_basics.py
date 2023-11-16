"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with the Amazon EMR API to create
and manage clusters and job steps.
"""
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

def run_job_flow(name, log_uri, keep_alive, applications, job_flow_role, service_role, security_groups, steps, emr_client):
    if False:
        return 10
    "\n    Runs a job flow with the specified steps. A job flow creates a cluster of\n    instances and adds steps to be run on the cluster. Steps added to the cluster\n    are run as soon as the cluster is ready.\n\n    This example uses the 'emr-5.30.1' release. A list of recent releases can be\n    found here:\n        https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-release-components.html.\n\n    :param name: The name of the cluster.\n    :param log_uri: The URI where logs are stored. This can be an Amazon S3 bucket URL,\n                    such as 's3://my-log-bucket'.\n    :param keep_alive: When True, the cluster is put into a Waiting state after all\n                       steps are run. When False, the cluster terminates itself when\n                       the step queue is empty.\n    :param applications: The applications to install on each instance in the cluster,\n                         such as Hive or Spark.\n    :param job_flow_role: The IAM role assumed by the cluster.\n    :param service_role: The IAM role assumed by the service.\n    :param security_groups: The security groups to assign to the cluster instances.\n                            Amazon EMR adds all needed rules to these groups, so\n                            they can be empty if you require only the default rules.\n    :param steps: The job flow steps to add to the cluster. These are run in order\n                  when the cluster is ready.\n    :param emr_client: The Boto3 EMR client object.\n    :return: The ID of the newly created cluster.\n    "
    try:
        response = emr_client.run_job_flow(Name=name, LogUri=log_uri, ReleaseLabel='emr-5.30.1', Instances={'MasterInstanceType': 'm5.xlarge', 'SlaveInstanceType': 'm5.xlarge', 'InstanceCount': 3, 'KeepJobFlowAliveWhenNoSteps': keep_alive, 'EmrManagedMasterSecurityGroup': security_groups['manager'].id, 'EmrManagedSlaveSecurityGroup': security_groups['worker'].id}, Steps=[{'Name': step['name'], 'ActionOnFailure': 'CONTINUE', 'HadoopJarStep': {'Jar': 'command-runner.jar', 'Args': ['spark-submit', '--deploy-mode', 'cluster', step['script_uri'], *step['script_args']]}} for step in steps], Applications=[{'Name': app} for app in applications], JobFlowRole=job_flow_role.name, ServiceRole=service_role.name, EbsRootVolumeSize=10, VisibleToAllUsers=True)
        cluster_id = response['JobFlowId']
        logger.info('Created cluster %s.', cluster_id)
    except ClientError:
        logger.exception("Couldn't create cluster.")
        raise
    else:
        return cluster_id

def describe_cluster(cluster_id, emr_client):
    if False:
        print('Hello World!')
    '\n    Gets detailed information about a cluster.\n\n    :param cluster_id: The ID of the cluster to describe.\n    :param emr_client: The Boto3 EMR client object.\n    :return: The retrieved cluster information.\n    '
    try:
        response = emr_client.describe_cluster(ClusterId=cluster_id)
        cluster = response['Cluster']
        logger.info('Got data for cluster %s.', cluster['Name'])
    except ClientError:
        logger.exception("Couldn't get data for cluster %s.", cluster_id)
        raise
    else:
        return cluster

def terminate_cluster(cluster_id, emr_client):
    if False:
        for i in range(10):
            print('nop')
    '\n    Terminates a cluster. This terminates all instances in the cluster and cannot\n    be undone. Any data not saved elsewhere, such as in an Amazon S3 bucket, is lost.\n\n    :param cluster_id: The ID of the cluster to terminate.\n    :param emr_client: The Boto3 EMR client object.\n    '
    try:
        emr_client.terminate_job_flows(JobFlowIds=[cluster_id])
        logger.info('Terminated cluster %s.', cluster_id)
    except ClientError:
        logger.exception("Couldn't terminate cluster %s.", cluster_id)
        raise

def add_step(cluster_id, name, script_uri, script_args, emr_client):
    if False:
        while True:
            i = 10
    '\n    Adds a job step to the specified cluster. This example adds a Spark\n    step, which is run by the cluster as soon as it is added.\n\n    :param cluster_id: The ID of the cluster.\n    :param name: The name of the step.\n    :param script_uri: The URI where the Python script is stored.\n    :param script_args: Arguments to pass to the Python script.\n    :param emr_client: The Boto3 EMR client object.\n    :return: The ID of the newly added step.\n    '
    try:
        response = emr_client.add_job_flow_steps(JobFlowId=cluster_id, Steps=[{'Name': name, 'ActionOnFailure': 'CONTINUE', 'HadoopJarStep': {'Jar': 'command-runner.jar', 'Args': ['spark-submit', '--deploy-mode', 'cluster', script_uri, *script_args]}}])
        step_id = response['StepIds'][0]
        logger.info('Started step with ID %s', step_id)
    except ClientError:
        logger.exception("Couldn't start step %s with URI %s.", name, script_uri)
        raise
    else:
        return step_id

def list_steps(cluster_id, emr_client):
    if False:
        print('Hello World!')
    '\n    Gets a list of steps for the specified cluster. In this example, all steps are\n    returned, including completed and failed steps.\n\n    :param cluster_id: The ID of the cluster.\n    :param emr_client: The Boto3 EMR client object.\n    :return: The list of steps for the specified cluster.\n    '
    try:
        response = emr_client.list_steps(ClusterId=cluster_id)
        steps = response['Steps']
        logger.info('Got %s steps for cluster %s.', len(steps), cluster_id)
    except ClientError:
        logger.exception("Couldn't get steps for cluster %s.", cluster_id)
        raise
    else:
        return steps

def describe_step(cluster_id, step_id, emr_client):
    if False:
        return 10
    '\n    Gets detailed information about the specified step, including the current state of\n    the step.\n\n    :param cluster_id: The ID of the cluster.\n    :param step_id: The ID of the step.\n    :param emr_client: The Boto3 EMR client object.\n    :return: The retrieved information about the specified step.\n    '
    try:
        response = emr_client.describe_step(ClusterId=cluster_id, StepId=step_id)
        step = response['Step']
        logger.info('Got data for step %s.', step_id)
    except ClientError:
        logger.exception("Couldn't get data for step %s.", step_id)
        raise
    else:
        return step