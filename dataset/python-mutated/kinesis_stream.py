"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Kinesis to create and
manage streams.
"""
import json
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class KinesisStream:
    """Encapsulates a Kinesis stream."""

    def __init__(self, kinesis_client):
        if False:
            while True:
                i = 10
        '\n        :param kinesis_client: A Boto3 Kinesis client.\n        '
        self.kinesis_client = kinesis_client
        self.name = None
        self.details = None
        self.stream_exists_waiter = kinesis_client.get_waiter('stream_exists')

    def _clear(self):
        if False:
            print('Hello World!')
        '\n        Clears property data of the stream object.\n        '
        self.name = None
        self.details = None

    def arn(self):
        if False:
            print('Hello World!')
        '\n        Gets the Amazon Resource Name (ARN) of the stream.\n        '
        return self.details['StreamARN']

    def create(self, name, wait_until_exists=True):
        if False:
            return 10
        '\n        Creates a stream.\n\n        :param name: The name of the stream.\n        :param wait_until_exists: When True, waits until the service reports that\n                                  the stream exists, then queries for its metadata.\n        '
        try:
            self.kinesis_client.create_stream(StreamName=name, ShardCount=1)
            self.name = name
            logger.info('Created stream %s.', name)
            if wait_until_exists:
                logger.info('Waiting until exists.')
                self.stream_exists_waiter.wait(StreamName=name)
                self.describe(name)
        except ClientError:
            logger.exception("Couldn't create stream %s.", name)
            raise

    def describe(self, name):
        if False:
            while True:
                i = 10
        '\n        Gets metadata about a stream.\n\n        :param name: The name of the stream.\n        :return: Metadata about the stream.\n        '
        try:
            response = self.kinesis_client.describe_stream(StreamName=name)
            self.name = name
            self.details = response['StreamDescription']
            logger.info('Got stream %s.', name)
        except ClientError:
            logger.exception("Couldn't get %s.", name)
            raise
        else:
            return self.details

    def delete(self):
        if False:
            i = 10
            return i + 15
        '\n        Deletes a stream.\n        '
        try:
            self.kinesis_client.delete_stream(StreamName=self.name)
            self._clear()
            logger.info('Deleted stream %s.', self.name)
        except ClientError:
            logger.exception("Couldn't delete stream %s.", self.name)
            raise

    def put_record(self, data, partition_key):
        if False:
            print('Hello World!')
        '\n        Puts data into the stream. The data is formatted as JSON before it is passed\n        to the stream.\n\n        :param data: The data to put in the stream.\n        :param partition_key: The partition key to use for the data.\n        :return: Metadata about the record, including its shard ID and sequence number.\n        '
        try:
            response = self.kinesis_client.put_record(StreamName=self.name, Data=json.dumps(data), PartitionKey=partition_key)
            logger.info('Put record in stream %s.', self.name)
        except ClientError:
            logger.exception("Couldn't put record in stream %s.", self.name)
            raise
        else:
            return response

    def get_records(self, max_records):
        if False:
            print('Hello World!')
        '\n        Gets records from the stream. This function is a generator that first gets\n        a shard iterator for the stream, then uses the shard iterator to get records\n        in batches from the stream. Each batch of records is yielded back to the\n        caller until the specified maximum number of records has been retrieved.\n\n        :param max_records: The maximum number of records to retrieve.\n        :return: Yields the current batch of retrieved records.\n        '
        try:
            response = self.kinesis_client.get_shard_iterator(StreamName=self.name, ShardId=self.details['Shards'][0]['ShardId'], ShardIteratorType='LATEST')
            shard_iter = response['ShardIterator']
            record_count = 0
            while record_count < max_records:
                response = self.kinesis_client.get_records(ShardIterator=shard_iter, Limit=10)
                shard_iter = response['NextShardIterator']
                records = response['Records']
                logger.info('Got %s records.', len(records))
                record_count += len(records)
                yield records
        except ClientError:
            logger.exception("Couldn't get records from stream %s.", self.name)
            raise