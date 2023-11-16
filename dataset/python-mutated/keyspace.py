import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class KeyspaceWrapper:
    """Encapsulates Amazon Keyspaces (for Apache Cassandra) keyspace and table actions."""

    def __init__(self, keyspaces_client):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param keyspaces_client: A Boto3 Amazon Keyspaces client.\n        '
        self.keyspaces_client = keyspaces_client
        self.ks_name = None
        self.ks_arn = None
        self.table_name = None

    @classmethod
    def from_client(cls):
        if False:
            for i in range(10):
                print('nop')
        keyspaces_client = boto3.client('keyspaces')
        return cls(keyspaces_client)

    def create_keyspace(self, name):
        if False:
            while True:
                i = 10
        '\n        Creates a keyspace.\n\n        :param name: The name to give the keyspace.\n        :return: The Amazon Resource Name (ARN) of the new keyspace.\n        '
        try:
            response = self.keyspaces_client.create_keyspace(keyspaceName=name)
            self.ks_name = name
            self.ks_arn = response['resourceArn']
        except ClientError as err:
            logger.error("Couldn't create %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return self.ks_arn

    def exists_keyspace(self, name):
        if False:
            while True:
                i = 10
        '\n        Checks whether a keyspace exists.\n\n        :param name: The name of the keyspace to look up.\n        :return: True when the keyspace exists. Otherwise, False.\n        '
        try:
            response = self.keyspaces_client.get_keyspace(keyspaceName=name)
            self.ks_name = response['keyspaceName']
            self.ks_arn = response['resourceArn']
            exists = True
        except ClientError as err:
            if err.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info('Keyspace %s does not exist.', name)
                exists = False
            else:
                logger.error("Couldn't verify %s exists. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        return exists

    def list_keyspaces(self, limit):
        if False:
            print('Hello World!')
        '\n        Lists the keyspaces in your account.\n\n        :param limit: The maximum number of keyspaces to list.\n        '
        try:
            ks_paginator = self.keyspaces_client.get_paginator('list_keyspaces')
            for page in ks_paginator.paginate(PaginationConfig={'MaxItems': limit}):
                for ks in page['keyspaces']:
                    print(ks['keyspaceName'])
                    print(f"\t{ks['resourceArn']}")
        except ClientError as err:
            logger.error("Couldn't list keyspaces. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def create_table(self, table_name):
        if False:
            print('Hello World!')
        '\n        Creates a table in the  keyspace.\n        The table is created with a schema for storing movie data\n        and has point-in-time recovery enabled.\n\n        :param table_name: The name to give the table.\n        :return: The ARN of the new table.\n        '
        try:
            response = self.keyspaces_client.create_table(keyspaceName=self.ks_name, tableName=table_name, schemaDefinition={'allColumns': [{'name': 'title', 'type': 'text'}, {'name': 'year', 'type': 'int'}, {'name': 'release_date', 'type': 'timestamp'}, {'name': 'plot', 'type': 'text'}], 'partitionKeys': [{'name': 'year'}, {'name': 'title'}]}, pointInTimeRecovery={'status': 'ENABLED'})
        except ClientError as err:
            logger.error("Couldn't create table %s. Here's why: %s: %s", table_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['resourceArn']

    def get_table(self, table_name):
        if False:
            while True:
                i = 10
        '\n        Gets data about a table in the keyspace.\n\n        :param table_name: The name of the table to look up.\n        :return: Data about the table.\n        '
        try:
            response = self.keyspaces_client.get_table(keyspaceName=self.ks_name, tableName=table_name)
            self.table_name = table_name
        except ClientError as err:
            if err.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info('Table %s does not exist.', table_name)
                self.table_name = None
                response = None
            else:
                logger.error("Couldn't verify %s exists. Here's why: %s: %s", table_name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        return response

    def list_tables(self):
        if False:
            while True:
                i = 10
        '\n        Lists the tables in the keyspace.\n        '
        try:
            table_paginator = self.keyspaces_client.get_paginator('list_tables')
            for page in table_paginator.paginate(keyspaceName=self.ks_name):
                for table in page['tables']:
                    print(table['tableName'])
                    print(f"\t{table['resourceArn']}")
        except ClientError as err:
            logger.error("Couldn't list tables in keyspace %s. Here's why: %s: %s", self.ks_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def update_table(self):
        if False:
            while True:
                i = 10
        '\n        Updates the schema of the table.\n\n        This example updates a table of movie data by adding a new column\n        that tracks whether the movie has been watched.\n        '
        try:
            self.keyspaces_client.update_table(keyspaceName=self.ks_name, tableName=self.table_name, addColumns=[{'name': 'watched', 'type': 'boolean'}])
        except ClientError as err:
            logger.error("Couldn't update table %s. Here's why: %s: %s", self.table_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def restore_table(self, restore_timestamp):
        if False:
            return 10
        '\n        Restores the table to a previous point in time. The table is restored\n        to a new table in the same keyspace.\n\n        :param restore_timestamp: The point in time to restore the table. This time\n                                  must be in UTC format.\n        :return: The name of the restored table.\n        '
        try:
            restored_table_name = f'{self.table_name}_restored'
            self.keyspaces_client.restore_table(sourceKeyspaceName=self.ks_name, sourceTableName=self.table_name, targetKeyspaceName=self.ks_name, targetTableName=restored_table_name, restoreTimestamp=restore_timestamp)
        except ClientError as err:
            logger.error("Couldn't restore table %s. Here's why: %s: %s", restore_timestamp, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return restored_table_name

    def delete_table(self):
        if False:
            while True:
                i = 10
        '\n        Deletes the table from the keyspace.\n        '
        try:
            self.keyspaces_client.delete_table(keyspaceName=self.ks_name, tableName=self.table_name)
            self.table_name = None
        except ClientError as err:
            logger.error("Couldn't delete table %s. Here's why: %s: %s", self.table_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def delete_keyspace(self):
        if False:
            print('Hello World!')
        '\n        Deletes the keyspace.\n        '
        try:
            self.keyspaces_client.delete_keyspace(keyspaceName=self.ks_name)
            self.ks_name = None
        except ClientError as err:
            logger.error("Couldn't delete keyspace %s. Here's why: %s: %s", self.ks_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise