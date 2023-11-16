"""
Stub functions that are used by the Amazon Kinesis unit tests.
"""
import datetime
import json
from test_tools.example_stubber import ExampleStubber

class KinesisStubber(ExampleStubber):
    """
    A class that implements stub functions used by Amazon Kinesis unit tests.

    The stubbed functions expect certain parameters to be passed to them as
    part of the tests, and raise errors if the parameters are not as expected.
    """

    def __init__(self, client, use_stubs=True):
        if False:
            while True:
                i = 10
        '\n        Initializes the object with a specific client and configures it for\n        stubbing or AWS passthrough.\n\n        :param client: A Boto3 Kinesis client.\n        :param use_stubs: When True, use stubs to intercept requests. Otherwise,\n                          pass requests through to AWS.\n        '
        super().__init__(client, use_stubs)

    def stub_create_stream(self, stream_name, error_code=None):
        if False:
            print('Hello World!')
        expected_params = {'StreamName': stream_name, 'ShardCount': 1}
        response = {}
        self._stub_bifurcator('create_stream', expected_params, response, error_code=error_code)

    def stub_describe_stream(self, stream_name, stream_arn, status, error_code=None):
        if False:
            for i in range(10):
                print('nop')
        expected_params = {'StreamName': stream_name}
        response = {'StreamDescription': {'StreamName': stream_name, 'StreamARN': stream_arn, 'StreamStatus': status, 'Shards': [], 'HasMoreShards': False, 'RetentionPeriodHours': 10, 'StreamCreationTimestamp': datetime.datetime.now(), 'EnhancedMonitoring': []}}
        self._stub_bifurcator('describe_stream', expected_params, response, error_code=error_code)

    def stub_delete_stream(self, stream_name, error_code=None):
        if False:
            i = 10
            return i + 15
        expected_params = {'StreamName': stream_name}
        response = {}
        self._stub_bifurcator('delete_stream', expected_params, response, error_code=error_code)

    def stub_put_record(self, stream, data, partition_key, error_code=None):
        if False:
            print('Hello World!')
        expected_params = {'StreamName': stream, 'Data': json.dumps(data), 'PartitionKey': partition_key}
        response = {'ShardId': 'test-id', 'SequenceNumber': 'test-number'}
        self._stub_bifurcator('put_record', expected_params, response, error_code=error_code)

    def stub_put_records(self, stream, batch, partition_key, error_code=None):
        if False:
            return 10
        expected_params = {'StreamName': stream, 'Records': [{'Data': json.dumps(record), 'PartitionKey': partition_key} for record in batch]}
        response = {'Records': [{'ShardId': 'test-id', 'SequenceNumber': 'test-number'}]}
        self._stub_bifurcator('put_records', expected_params, response, error_code=error_code)

    def stub_get_shard_iterator(self, stream_name, shard_id, shard_iter, error_code=None):
        if False:
            print('Hello World!')
        expected_params = {'StreamName': stream_name, 'ShardId': shard_id, 'ShardIteratorType': 'LATEST'}
        response = {'ShardIterator': shard_iter}
        self._stub_bifurcator('get_shard_iterator', expected_params, response, error_code=error_code)

    def stub_get_records(self, shard_iter, limit, records, error_code=None):
        if False:
            for i in range(10):
                print('nop')
        expected_params = {'ShardIterator': shard_iter, 'Limit': limit}
        response = {'NextShardIterator': shard_iter, 'Records': [{'Data': record, 'SequenceNumber': '1', 'PartitionKey': 'partition_key'} for record in records]}
        self._stub_bifurcator('get_records', expected_params, response, error_code=error_code)