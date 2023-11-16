"""
Purpose

Shows how to run an elastic map reduce file system (EMRFS) command as a job step on
an Amazon EMR cluster. This can be used to automate EMRFS commands and is an
alternative to connecting through SSH to run the commands manually.
"""
import boto3
from botocore.exceptions import ClientError

def add_emrfs_step(command, bucket_url, cluster_id, emr_client):
    if False:
        i = 10
        return i + 15
    '\n    Add an EMRFS command as a job flow step to an existing cluster.\n\n    :param command: The EMRFS command to run.\n    :param bucket_url: The URL of a bucket that contains tracking metadata.\n    :param cluster_id: The ID of the cluster to update.\n    :param emr_client: The Boto3 Amazon EMR client object.\n    :return: The ID of the added job flow step. Status can be tracked by calling\n             the emr_client.describe_step() function.\n    '
    job_flow_step = {'Name': 'Example EMRFS Command Step', 'ActionOnFailure': 'CONTINUE', 'HadoopJarStep': {'Jar': 'command-runner.jar', 'Args': ['/usr/bin/emrfs', command, bucket_url]}}
    try:
        response = emr_client.add_job_flow_steps(JobFlowId=cluster_id, Steps=[job_flow_step])
        step_id = response['StepIds'][0]
        print(f'Added step {step_id} to cluster {cluster_id}.')
    except ClientError:
        print(f"Couldn't add a step to cluster {cluster_id}.")
        raise
    else:
        return step_id

def usage_demo():
    if False:
        for i in range(10):
            print('nop')
    emr_client = boto3.client('emr')
    cluster = emr_client.list_clusters(ClusterStates=['WAITING'])['Clusters'][0]
    add_emrfs_step('sync', 's3://elasticmapreduce/samples/cloudfront', cluster['Id'], emr_client)
if __name__ == '__main__':
    usage_demo()