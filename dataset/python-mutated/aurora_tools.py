"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) to create and delete an Amazon Aurora
(Aurora) database cluster and create and delete an AWS Secrets Manager secret.
"""
import json
import logging
import sys
from botocore.exceptions import ClientError
sys.path.append('../..')
from demo_tools.custom_waiter import CustomWaiter, WaitState
logger = logging.getLogger(__name__)

class ClusterAvailableWaiter(CustomWaiter):
    """
    Waits for the database cluster to be available.
    """

    def __init__(self, client):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('ClusterAvailable', 'DescribeDBClusters', 'DBClusters[].Status', {'available': WaitState.SUCCESS}, client, matcher='pathAny')

    def wait(self, cluster_name):
        if False:
            print('Hello World!')
        self._wait(DBClusterIdentifier=cluster_name)

def create_db_cluster(cluster_name, db_name, admin_name, admin_password, rds_client):
    if False:
        while True:
            i = 10
    '\n    Creates a serverless Amazon Aurora database cluster and a MySQL database\n    within it.\n\n    :param cluster_name: The name of the cluster to create.\n    :param db_name: The name of the database to create.\n    :param admin_name: The username of the database administrator.\n    :param admin_password: The password of the database administrator.\n    :param rds_client: The Boto3 Amazon RDS client.\n    :return: The newly created cluster.\n    '
    try:
        response = rds_client.create_db_cluster(DatabaseName=db_name, DBClusterIdentifier=cluster_name, Engine='aurora-mysql', EngineMode='serverless', MasterUsername=admin_name, MasterUserPassword=admin_password, EnableHttpEndpoint=True)
        cluster = response['DBCluster']
        logger.info('Created database %s in cluster %s.', cluster['DatabaseName'], cluster['DBClusterIdentifier'])
    except ClientError:
        logger.exception("Couldn't create database %s.", db_name)
        raise
    else:
        return cluster

def delete_db_cluster(cluster_name, rds_client):
    if False:
        while True:
            i = 10
    '\n    Deletes an Amazon Aurora cluster.\n\n    :param cluster_name: The name of the cluster to delete.\n    :param rds_client: The Boto3 Amazon RDS client.\n    '
    try:
        rds_client.delete_db_cluster(DBClusterIdentifier=cluster_name, SkipFinalSnapshot=True)
        logger.info('Deleted cluster %s.', cluster_name)
    except ClientError:
        logger.exception("Couldn't delete cluster %s.", cluster_name)
        raise

def create_aurora_secret(secret_name, username, password, engine, host, port, cluster_name, secrets_client):
    if False:
        return 10
    '\n    Creates an AWS Secrets Manager secret that contains MySQL user credentials.\n\n    :param secret_name: The name of the secret to create.\n    :param username: The username to store in the credentials.\n    :param password: The password to store in the credentials.\n    :param engine: The database engine these credentials are for, such as MySQL.\n    :param host: The endpoint URL of the Aurora cluster that contains the database\n                 these credentials are for.\n    :param port: The port that can be used to connect to the database endpoint.\n    :param cluster_name: The name of the cluster that contains the database.\n    :param secrets_client: The Boto3 Secrets Manager client.\n    :return The newly created secret.\n    '
    aurora_admin_secret = {'username': username, 'password': password, 'engine': engine, 'host': host, 'port': port, 'dbClusterIdentifier': cluster_name}
    try:
        secret = secrets_client.create_secret(Name=secret_name, SecretString=json.dumps(aurora_admin_secret))
        logger.info('Created secret %s.', secret_name)
    except ClientError:
        logger.exception("Couldn't create secret %s.", secret_name)
        raise
    else:
        return secret

def delete_secret(secret_name, secrets_client):
    if False:
        while True:
            i = 10
    '\n    Deletes an AWS Secrets Manager secret. Recovery data is not saved, so after\n    this action completes the secret cannot be used or recovered.\n\n    :param secret_name: The name of the secret to delete.\n    :param secrets_client: The Boto3 Secrets Manager client.\n    '
    try:
        secrets_client.delete_secret(SecretId=secret_name, ForceDeleteWithoutRecovery=True)
        logger.info('Deleted secret %s.', secret_name)
    except ClientError:
        logger.exception("Couldn't delete secret %s.", secret_name)
        raise