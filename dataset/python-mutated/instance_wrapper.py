"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) to create and manage Amazon Relational
Database Service (Amazon RDS) DB instances.
"""
import json
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class InstanceWrapper:
    """Encapsulates Amazon RDS DB instance actions."""

    def __init__(self, rds_client):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param rds_client: A Boto3 Amazon RDS client.\n        '
        self.rds_client = rds_client

    @classmethod
    def from_client(cls):
        if False:
            i = 10
            return i + 15
        '\n        Instantiates this class from a Boto3 client.\n        '
        rds_client = boto3.client('rds')
        return cls(rds_client)

    def get_parameter_group(self, parameter_group_name):
        if False:
            i = 10
            return i + 15
        '\n        Gets a DB parameter group.\n\n        :param parameter_group_name: The name of the parameter group to retrieve.\n        :return: The parameter group.\n        '
        try:
            response = self.rds_client.describe_db_parameter_groups(DBParameterGroupName=parameter_group_name)
            parameter_group = response['DBParameterGroups'][0]
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
        '\n        Creates a DB parameter group that is based on the specified parameter group\n        family.\n\n        :param parameter_group_name: The name of the newly created parameter group.\n        :param parameter_group_family: The family that is used as the basis of the new\n                                       parameter group.\n        :param description: A description given to the parameter group.\n        :return: Data about the newly created parameter group.\n        '
        try:
            response = self.rds_client.create_db_parameter_group(DBParameterGroupName=parameter_group_name, DBParameterGroupFamily=parameter_group_family, Description=description)
        except ClientError as err:
            logger.error("Couldn't create parameter group %s. Here's why: %s: %s", parameter_group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def delete_parameter_group(self, parameter_group_name):
        if False:
            return 10
        '\n        Deletes a DB parameter group.\n\n        :param parameter_group_name: The name of the parameter group to delete.\n        :return: Data about the parameter group.\n        '
        try:
            self.rds_client.delete_db_parameter_group(DBParameterGroupName=parameter_group_name)
        except ClientError as err:
            logger.error("Couldn't delete parameter group %s. Here's why: %s: %s", parameter_group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def get_parameters(self, parameter_group_name, name_prefix='', source=None):
        if False:
            print('Hello World!')
        "\n        Gets the parameters that are contained in a DB parameter group.\n\n        :param parameter_group_name: The name of the parameter group to query.\n        :param name_prefix: When specified, the retrieved list of parameters is filtered\n                            to contain only parameters that start with this prefix.\n        :param source: When specified, only parameters from this source are retrieved.\n                       For example, a source of 'user' retrieves only parameters that\n                       were set by a user.\n        :return: The list of requested parameters.\n        "
        try:
            kwargs = {'DBParameterGroupName': parameter_group_name}
            if source is not None:
                kwargs['Source'] = source
            parameters = []
            paginator = self.rds_client.get_paginator('describe_db_parameters')
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
        '\n        Updates parameters in a custom DB parameter group.\n\n        :param parameter_group_name: The name of the parameter group to update.\n        :param update_parameters: The parameters to update in the group.\n        :return: Data about the modified parameter group.\n        '
        try:
            response = self.rds_client.modify_db_parameter_group(DBParameterGroupName=parameter_group_name, Parameters=update_parameters)
        except ClientError as err:
            logger.error("Couldn't update parameters in %s. Here's why: %s: %s", parameter_group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def create_snapshot(self, snapshot_id, instance_id):
        if False:
            return 10
        '\n        Creates a snapshot of a DB instance.\n\n        :param snapshot_id: The ID to give the created snapshot.\n        :param instance_id: The ID of the DB instance to snapshot.\n        :return: Data about the newly created snapshot.\n        '
        try:
            response = self.rds_client.create_db_snapshot(DBSnapshotIdentifier=snapshot_id, DBInstanceIdentifier=instance_id)
            snapshot = response['DBSnapshot']
        except ClientError as err:
            logger.error("Couldn't create snapshot of %s. Here's why: %s: %s", instance_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return snapshot

    def get_snapshot(self, snapshot_id):
        if False:
            i = 10
            return i + 15
        '\n        Gets a DB instance snapshot.\n\n        :param snapshot_id: The ID of the snapshot to retrieve.\n        :return: The retrieved snapshot.\n        '
        try:
            response = self.rds_client.describe_db_snapshots(DBSnapshotIdentifier=snapshot_id)
            snapshot = response['DBSnapshots'][0]
        except ClientError as err:
            logger.error("Couldn't get snapshot %s. Here's why: %s: %s", snapshot_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return snapshot

    def get_engine_versions(self, engine, parameter_group_family=None):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
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

    def create_db_instance(self, db_name, instance_id, parameter_group_name, db_engine, db_engine_version, instance_class, storage_type, allocated_storage, admin_name, admin_password):
        if False:
            print('Hello World!')
        '\n        Creates a DB instance.\n\n        :param db_name: The name of the database that is created in the DB instance.\n        :param instance_id: The ID to give the newly created DB instance.\n        :param parameter_group_name: A parameter group to associate with the DB instance.\n        :param db_engine: The database engine of a database to create in the DB instance.\n        :param db_engine_version: The engine version for the created database.\n        :param instance_class: The DB instance class for the newly created DB instance.\n        :param storage_type: The storage type of the DB instance.\n        :param allocated_storage: The amount of storage allocated on the DB instance, in GiBs.\n        :param admin_name: The name of the admin user for the created database.\n        :param admin_password: The admin password for the created database.\n        :return: Data about the newly created DB instance.\n        '
        try:
            response = self.rds_client.create_db_instance(DBName=db_name, DBInstanceIdentifier=instance_id, DBParameterGroupName=parameter_group_name, Engine=db_engine, EngineVersion=db_engine_version, DBInstanceClass=instance_class, StorageType=storage_type, AllocatedStorage=allocated_storage, MasterUsername=admin_name, MasterUserPassword=admin_password)
            db_inst = response['DBInstance']
        except ClientError as err:
            logger.error("Couldn't create DB instance %s. Here's why: %s: %s", instance_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return db_inst

    def delete_db_instance(self, instance_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes a DB instance.\n\n        :param instance_id: The ID of the DB instance to delete.\n        :return: Data about the deleted DB instance.\n        '
        try:
            response = self.rds_client.delete_db_instance(DBInstanceIdentifier=instance_id, SkipFinalSnapshot=True, DeleteAutomatedBackups=True)
            db_inst = response['DBInstance']
        except ClientError as err:
            logger.error("Couldn't delete DB instance %s. Here's why: %s: %s", instance_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return db_inst