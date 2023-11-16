"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) to write and retrieve Amazon DynamoDB
data as part of a REST API. This file is uploaded to AWS Lambda as part of the
serverless deployment package created by AWS Chalice.
"""
import datetime
import os
import random
import boto3
from boto3.dynamodb.conditions import Key

class Storage:
    """
    Handles basic storage functions, backed by an Amazon DynamoDB table.
    """
    STATES = {'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'}

    def __init__(self, table):
        if False:
            for i in range(10):
                print('nop')
        self._table = table

    @classmethod
    def from_env(cls):
        if False:
            return 10
        '\n        Creates a Storage object that contains a table identified by the TABLE_NAME\n        environment variable.\n\n        :return: The newly created Storage object.\n        '
        table_name = os.environ.get('TABLE_NAME', '')
        table = boto3.resource('dynamodb').Table(table_name)
        return cls(table)

    @staticmethod
    def _generate_random_data(state):
        if False:
            while True:
                i = 10
        '\n        Generates some random data for the demo.\n\n        :param state: The state for which to create the data.\n        :return: The newly created data.\n        '
        return {'state': state, 'date': datetime.date.today().isoformat(), 'cases': random.randint(1, 1000), 'deaths': random.randint(1, 100)}

    def get_state_data(self, state):
        if False:
            i = 10
            return i + 15
        "\n        Gets the data records for the specified state. If there are no records,\n        a new one is generated with random values for today's date and stored in\n        the table before it is returned.\n\n        :param state: The state to retrieve.\n        :return: The retrieved data.\n        "
        response = self._table.query(KeyConditionExpression=Key('state').eq(state))
        items = response.get('Items', [])
        if len(items) == 0:
            items.append(self._generate_random_data(state))
            self._table.put_item(Item=items[0])
        return items

    def put_state_data(self, state, state_data):
        if False:
            print('Hello World!')
        '\n        Puts data for a state into the table.\n\n        :param state: The state for which to store the data.\n        :param state_data: The data record to store.\n        '
        self._table.put_item(Item=state_data)

    def delete_state_data(self, state):
        if False:
            return 10
        '\n        Deletes all records for a state from the table.\n\n        :param state: The state to delete.\n        '
        response = self._table.query(KeyConditionExpression=Key('state').eq(state))
        items = response.get('Items', [])
        with self._table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={'state': item['state'], 'date': item['date']})

    def post_state_data(self, state, state_data):
        if False:
            print('Hello World!')
        '\n        Puts data for a state into the table.\n\n        :param state: The state for which to store the data.\n        :param state_data: The data record to store.\n        '
        self._table.put_item(Item=state_data)

    def get_state_date_data(self, state, date):
        if False:
            while True:
                i = 10
        '\n        Gets a single record for the specified state and date.\n\n        :param state: The state of the record to retrieve.\n        :param date: The date of the record to retrieve.\n        :return: The retrieved record, or None if no record exists.\n        '
        response = self._table.get_item(Key={'state': state, 'date': date})
        item = response.get('Item', None)
        return item

    def delete_state_date_data(self, state, date):
        if False:
            while True:
                i = 10
        '\n        Deletes the record for the specified state and date.\n\n        :param state: The state of the record to remove.\n        :param date: The date of the record to remove.\n        '
        self._table.delete_item(Key={'state': state, 'date': date})