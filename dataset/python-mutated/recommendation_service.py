import json
import logging
import boto3
from botocore.exceptions import ClientError
log = logging.getLogger(__name__)

class RecommendationServiceError(Exception):

    def __init__(self, table_name, message):
        if False:
            print('Hello World!')
        self.table_name = table_name
        self.message = message
        super().__init__(self.message)

class RecommendationService:
    """
    Encapsulates a DynamoDB table to use as a service that recommends books, movies,
    and songs.
    """

    def __init__(self, table_name, dynamodb_client):
        if False:
            i = 10
            return i + 15
        '\n        :param table_name: The name of the DynamoDB recommendations table.\n        :param dynamodb_client: A Boto3 DynamoDB client.\n        '
        self.table_name = table_name
        self.dynamodb_client = dynamodb_client

    @classmethod
    def from_client(cls, table_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates this class from a Boto3 client.\n\n        :param table_name: The name of the DynamoDB recommendations table.\n        '
        ddb_client = boto3.client('dynamodb')
        return cls(table_name, ddb_client)

    def create(self):
        if False:
            while True:
                i = 10
        "\n        Creates a DynamoDB table to use a recommendation service. The table has a\n        hash key named 'MediaType' that defines the type of media recommended, such as\n        Book or Movie, and a range key named 'ItemId' that, combined with the MediaType,\n        forms a unique identifier for the recommended item.\n\n        :return: Data about the newly created table.\n        "
        try:
            response = self.dynamodb_client.create_table(TableName=self.table_name, AttributeDefinitions=[{'AttributeName': 'MediaType', 'AttributeType': 'S'}, {'AttributeName': 'ItemId', 'AttributeType': 'N'}], KeySchema=[{'AttributeName': 'MediaType', 'KeyType': 'HASH'}, {'AttributeName': 'ItemId', 'KeyType': 'RANGE'}], ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5})
            log.info('Creating table %s...', self.table_name)
            waiter = self.dynamodb_client.get_waiter('table_exists')
            waiter.wait(TableName=self.table_name)
            log.info('Table %s created.', self.table_name)
        except ClientError as err:
            if err.response['Error']['Code'] == 'ResourceInUseException':
                log.info('Table %s exists, nothing to be do.', self.table_name)
            else:
                raise RecommendationServiceError(self.table_name, f'ClientError when creating table: {err}.')
        else:
            return response

    def populate(self, data_file):
        if False:
            while True:
                i = 10
        '\n        Populates the recommendations table from a JSON file.\n\n        :param data_file: The path to the data file.\n        '
        try:
            with open(data_file) as data:
                items = json.load(data)
            batch = [{'PutRequest': {'Item': item}} for item in items]
            self.dynamodb_client.batch_write_item(RequestItems={self.table_name: batch})
            log.info('Populated table %s with items from %s.', self.table_name, data_file)
        except ClientError as err:
            raise RecommendationServiceError(self.table_name, f"Couldn't populate table from {data_file}: {err}")

    def destroy(self):
        if False:
            print('Hello World!')
        '\n        Deletes the recommendations table.\n        '
        try:
            self.dynamodb_client.delete_table(TableName=self.table_name)
            log.info('Deleting table %s...', self.table_name)
            waiter = self.dynamodb_client.get_waiter('table_not_exists')
            waiter.wait(TableName=self.table_name)
            log.info('Table %s deleted.', self.table_name)
        except ClientError as err:
            if err.response['Error']['Code'] == 'ResourceNotFoundException':
                log.info('Table %s does not exist, nothing to do.', self.table_name)
            else:
                raise RecommendationServiceError(self.table_name, f'ClientError when deleting table: {err}.')