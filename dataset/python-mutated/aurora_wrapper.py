"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) to create and manage Amazon Aurora
DB clusters.
"""
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class AuroraWrapper:
    """Encapsulates Aurora DB cluster actions."""

    def __init__(self, rds_client):
        if False:
            while True:
                i = 10
        '\n        :param rds_client: A Boto3 Amazon Relational Database Service (Amazon RDS) client.\n        '
        self.rds_client = rds_client

    @classmethod
    def from_client(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Instantiates this class from a Boto3 client.\n        '
        rds_client = boto3.client('rds')
        return cls(rds_client)

    def get_parameter_group(self, parameter_group_name):
        if False:
            return 10
        '\n        Gets a DB cluster parameter group.\n\n        :param parameter_group_name: The name of the parameter group to retrieve.\n        :return: The requested parameter group.\n        '
        try:
            response = self.rds_client.describe_db_cluster_parameter_groups(DBClusterParameterGroupName=parameter_group_name)
            parameter_group = response['DBClusterParameterGroups'][0]
        except ClientError as err:
            if err.response['Error']['Code'] == 'DBParameterGroupNotFound':
                logger.info('Parameter group %s does not exist.', parameter_group_name)
            else:
                logger.error("Couldn't get parameter group %s. Here's why: %s: %s", parameter_group_name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return parameter_group

    def create_parameter_group(self, parameter_group_name, parameter_group_family, description):
        if False:
            return 10
        '\n        Creates a DB cluster parameter group that is based on the specified parameter group\n        family.\n\n        :param parameter_group_name: The name of the newly created parameter group.\n        :param parameter_group_family: The family that is used as the basis of the new\n                                       parameter group.\n        :param description: A description given to the parameter group.\n        :return: Data about the newly created parameter group.\n        '
        try:
            response = self.rds_client.create_db_cluster_parameter_group(DBClusterParameterGroupName=parameter_group_name, DBParameterGroupFamily=parameter_group_family, Description=description)
        except ClientError as err:
            logger.error("Couldn't create parameter group %s. Here's why: %s: %s", parameter_group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def delete_parameter_group(self, parameter_group_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes a DB cluster parameter group.\n\n        :param parameter_group_name: The name of the parameter group to delete.\n        :return: Data about the parameter group.\n        '
        try:
            response = self.rds_client.delete_db_cluster_parameter_group(DBClusterParameterGroupName=parameter_group_name)
        except ClientError as err:
            logger.error("Couldn't delete parameter group %s. Here's why: %s: %s", parameter_group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def get_parameters(self, parameter_group_name, name_prefix='', source=None):
        if False:
            return 10
        "\n        Gets the parameters that are contained in a DB cluster parameter group.\n\n        :param parameter_group_name: The name of the parameter group to query.\n        :param name_prefix: When specified, the retrieved list of parameters is filtered\n                            to contain only parameters that start with this prefix.\n        :param source: When specified, only parameters from this source are retrieved.\n                       For example, a source of 'user' retrieves only parameters that\n                       were set by a user.\n        :return: The list of requested parameters.\n        "
        try:
            kwargs = {'DBClusterParameterGroupName': parameter_group_name}
            if source is not None:
                kwargs['Source'] = source
            parameters = []
            paginator = self.rds_client.get_paginator('describe_db_cluster_parameters')
            for page in paginator.paginate(**kwargs):
                parameters += [p for p in page['Parameters'] if p['ParameterName'].startswith(name_prefix)]
        except ClientError as err:
            logger.error("Couldn't get parameters for %s. Here's why: %s: %s", parameter_group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return parameters

    def update_parameters(self, parameter_group_name, update_parameters):
        if False:
            while True:
                i = 10
        '\n        Updates parameters in a custom DB cluster parameter group.\n\n        :param parameter_group_name: The name of the parameter group to update.\n        :param update_parameters: The parameters to update in the group.\n        :return: Data about the modified parameter group.\n        '
        try:
            response = self.rds_client.modify_db_cluster_parameter_group(DBClusterParameterGroupName=parameter_group_name, Parameters=update_parameters)
        except ClientError as err:
            logger.error("Couldn't update parameters in %s. Here's why: %s: %s", parameter_group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def get_db_cluster(self, cluster_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets data about an Aurora DB cluster.\n\n        :param cluster_name: The name of the DB cluster to retrieve.\n        :return: The retrieved DB cluster.\n        '
        try:
            response = self.rds_client.describe_db_clusters(DBClusterIdentifier=cluster_name)
            cluster = response['DBClusters'][0]
        except ClientError as err:
            if err.response['Error']['Code'] == 'DBClusterNotFoundFault':
                logger.info('Cluster %s does not exist.', cluster_name)
            else:
                logger.error("Couldn't verify the existence of DB cluster %s. Here's why: %s: %s", cluster_name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return cluster

    def create_db_cluster(self, cluster_name, parameter_group_name, db_name, db_engine, db_engine_version, admin_name, admin_password):
        if False:
            i = 10
            return i + 15
        '\n        Creates a DB cluster that is configured to use the specified parameter group.\n        The newly created DB cluster contains a database that uses the specified engine and\n        engine version.\n\n        :param cluster_name: The name of the DB cluster to create.\n        :param parameter_group_name: The name of the parameter group to associate with\n                                     the DB cluster.\n        :param db_name: The name of the database to create.\n        :param db_engine: The database engine of the database that is created, such as MySql.\n        :param db_engine_version: The version of the database engine.\n        :param admin_name: The user name of the database administrator.\n        :param admin_password: The password of the database administrator.\n        :return: The newly created DB cluster.\n        '
        try:
            response = self.rds_client.create_db_cluster(DatabaseName=db_name, DBClusterIdentifier=cluster_name, DBClusterParameterGroupName=parameter_group_name, Engine=db_engine, EngineVersion=db_engine_version, MasterUsername=admin_name, MasterUserPassword=admin_password)
            cluster = response['DBCluster']
        except ClientError as err:
            logger.error("Couldn't create database %s. Here's why: %s: %s", db_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return cluster

    def delete_db_cluster(self, cluster_name):
        if False:
            return 10
        '\n        Deletes a DB cluster.\n\n        :param cluster_name: The name of the DB cluster to delete.\n        '
        try:
            self.rds_client.delete_db_cluster(DBClusterIdentifier=cluster_name, SkipFinalSnapshot=True)
            logger.info('Deleted DB cluster %s.', cluster_name)
        except ClientError:
            logger.exception("Couldn't delete DB cluster %s.", cluster_name)
            raise

    def create_cluster_snapshot(self, snapshot_id, cluster_id):
        if False:
            while True:
                i = 10
        '\n        Creates a snapshot of a DB cluster.\n\n        :param snapshot_id: The ID to give the created snapshot.\n        :param cluster_id: The DB cluster to snapshot.\n        :return: Data about the newly created snapshot.\n        '
        try:
            response = self.rds_client.create_db_cluster_snapshot(DBClusterSnapshotIdentifier=snapshot_id, DBClusterIdentifier=cluster_id)
            snapshot = response['DBClusterSnapshot']
        except ClientError as err:
            logger.error("Couldn't create snapshot of %s. Here's why: %s: %s", cluster_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return snapshot

    def get_cluster_snapshot(self, snapshot_id):
        if False:
            return 10
        '\n        Gets a DB cluster snapshot.\n\n        :param snapshot_id: The ID of the snapshot to retrieve.\n        :return: The retrieved snapshot.\n        '
        try:
            response = self.rds_client.describe_db_cluster_snapshots(DBClusterSnapshotIdentifier=snapshot_id)
            snapshot = response['DBClusterSnapshots'][0]
        except ClientError as err:
            logger.error("Couldn't get DB cluster snapshot %s. Here's why: %s: %s", snapshot_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return snapshot

    def create_instance_in_cluster(self, instance_id, cluster_id, db_engine, instance_class):
        if False:
            return 10
        '\n        Creates a database instance in an existing DB cluster. The first database that is\n        created defaults to a read-write DB instance.\n\n        :param instance_id: The ID to give the newly created DB instance.\n        :param cluster_id: The ID of the DB cluster where the DB instance is created.\n        :param db_engine: The database engine of a database to create in the DB instance.\n                          This must be compatible with the configured parameter group\n                          of the DB cluster.\n        :param instance_class: The DB instance class for the newly created DB instance.\n        :return: Data about the newly created DB instance.\n        '
        try:
            response = self.rds_client.create_db_instance(DBInstanceIdentifier=instance_id, DBClusterIdentifier=cluster_id, Engine=db_engine, DBInstanceClass=instance_class)
            db_inst = response['DBInstance']
        except ClientError as err:
            logger.error("Couldn't create DB instance %s. Here's why: %s: %s", instance_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return db_inst

    def get_engine_versions(self, engine, parameter_group_family=None):
        if False:
            i = 10
            return i + 15
        '\n        Gets database engine versions that are available for the specified engine\n        and parameter group family.\n\n        :param engine: The database engine to look up.\n        :param parameter_group_family: When specified, restricts the returned list of\n                                       engine versions to those that are compatible with\n                                       this parameter group family.\n        :return: The list of database engine versions.\n        '
        try:
            kwargs = {'Engine': engine}
            if parameter_group_family is not None:
                kwargs['DBParameterGroupFamily'] = parameter_group_family
            response = self.rds_client.describe_db_engine_versions(**kwargs)
            versions = response['DBEngineVersions']
        except ClientError as err:
            logger.error("Couldn't get engine versions for %s. Here's why: %s: %s", engine, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return versions

    def get_orderable_instances(self, db_engine, db_engine_version):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets DB instance options that can be used to create DB instances that are\n        compatible with a set of specifications.\n\n        :param db_engine: The database engine that must be supported by the DB instance.\n        :param db_engine_version: The engine version that must be supported by the DB instance.\n        :return: The list of DB instance options that can be used to create a compatible DB instance.\n        '
        try:
            inst_opts = []
            paginator = self.rds_client.get_paginator('describe_orderable_db_instance_options')
            for page in paginator.paginate(Engine=db_engine, EngineVersion=db_engine_version):
                inst_opts += page['OrderableDBInstanceOptions']
        except ClientError as err:
            logger.error("Couldn't get orderable DB instances. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return inst_opts

    def get_db_instance(self, instance_id):
        if False:
            while True:
                i = 10
        '\n        Gets data about a DB instance.\n\n        :param instance_id: The ID of the DB instance to retrieve.\n        :return: The retrieved DB instance.\n        '
        try:
            response = self.rds_client.describe_db_instances(DBInstanceIdentifier=instance_id)
            db_inst = response['DBInstances'][0]
        except ClientError as err:
            if err.response['Error']['Code'] == 'DBInstanceNotFound':
                logger.info('Instance %s does not exist.', instance_id)
            else:
                logger.error("Couldn't get DB instance %s. Here's why: %s: %s", instance_id, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return db_inst

    def delete_db_instance(self, instance_id):
        if False:
            print('Hello World!')
        '\n        Deletes a DB instance.\n\n        :param instance_id: The ID of the DB instance to delete.\n        :return: Data about the deleted DB instance.\n        '
        try:
            response = self.rds_client.delete_db_instance(DBInstanceIdentifier=instance_id, SkipFinalSnapshot=True, DeleteAutomatedBackups=True)
            db_inst = response['DBInstance']
        except ClientError as err:
            logger.error("Couldn't delete DB instance %s. Here's why: %s: %s", instance_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return db_inst