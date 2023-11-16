import json
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Protocol
import pytest
import requests
from boto3.s3.transfer import KB, TransferConfig
from botocore.exceptions import ClientError
from localstack.constants import SECONDARY_TEST_AWS_REGION_NAME
from localstack.testing.aws.util import is_aws_cloud
from localstack.testing.pytest import markers
from localstack.utils.aws import arns
from localstack.utils.strings import short_uid
from localstack.utils.sync import retry
if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.literals import EventType
    from mypy_boto3_sqs import SQSClient
LOG = logging.getLogger(__name__)

class NotificationFactory(Protocol):
    """
    A protocol for connecting a bucket to a queue with a notification configurations and the necessary policies.
    """

    def __call__(self, bucket_name: str, queue_url: str, events: List['EventType']) -> None:
        if False:
            print('Hello World!')
        '\n        Creates a new notification configuration and respective policies.\n\n        :param bucket_name: the source bucket\n        :param queue_url: the target SQS queue\n        :param events: the type of S3 events to trigger the notification\n        :return: None\n        '
        raise NotImplementedError

def get_queue_arn(sqs_client, queue_url: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the given Queue's ARN. Expects the Queue to exist.\n\n    :param sqs_client: the boto3 client\n    :param queue_url: the queue URL\n    :return: the QueueARN\n    "
    response = sqs_client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['QueueArn'])
    return response['Attributes']['QueueArn']

def set_policy_for_queue(sqs_client, queue_url, bucket_name):
    if False:
        for i in range(10):
            print('nop')
    queue_arn = get_queue_arn(sqs_client, queue_url)
    assert queue_arn
    bucket_arn = arns.s3_bucket_arn(bucket_name)
    policy = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': '*', 'Action': 'sqs:SendMessage', 'Resource': queue_arn, 'Condition': {'ArnEquals': {'aws:SourceArn': bucket_arn}}}]}
    sqs_client.set_queue_attributes(QueueUrl=queue_url, Attributes={'Policy': json.dumps(policy)})
    return queue_arn

def create_sqs_bucket_notification(s3_client: 'S3Client', sqs_client: 'SQSClient', bucket_name: str, queue_url: str, events: List['EventType']):
    if False:
        i = 10
        return i + 15
    'A NotificationFactory.'
    queue_arn = set_policy_for_queue(sqs_client, queue_url, bucket_name)
    s3_client.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=dict(QueueConfigurations=[dict(QueueArn=queue_arn, Events=events)]))

@pytest.fixture
def s3_create_sqs_bucket_notification(aws_client) -> NotificationFactory:
    if False:
        while True:
            i = 10
    '\n    A factory fixture for creating sqs bucket notifications.\n    '

    def factory(bucket_name: str, queue_url: str, events: List['EventType'], s3_client=aws_client.s3, sqs_client=aws_client.sqs):
        if False:
            i = 10
            return i + 15
        return create_sqs_bucket_notification(s3_client, sqs_client, bucket_name, queue_url, events)
    return factory

def sqs_collect_s3_events(sqs_client: 'SQSClient', queue_url: str, min_events: int, timeout: int=10) -> List[Dict]:
    if False:
        while True:
            i = 10
    '\n    Polls the given queue for the given amount of time and extracts and flattens from the received messages all\n    events (messages that have a "Records" field in their body, and where the records can be json-deserialized).\n\n    :param sqs_client: the boto3 client to use\n    :param queue_url: the queue URL to listen from\n    :param min_events: the minimum number of events to receive to wait for\n    :param timeout: the number of seconds to wait before raising an assert error\n    :return: a list with the deserialized records from the SQS messages\n    '
    events = []

    def collect_events() -> None:
        if False:
            for i in range(10):
                print('nop')
        _response = sqs_client.receive_message(QueueUrl=queue_url, WaitTimeSeconds=1, MaxNumberOfMessages=1)
        messages = _response.get('Messages', [])
        if not messages:
            LOG.info('no messages received from %s after 1 second', queue_url)
        for m in messages:
            body = m['Body']
            if 's3:TestEvent' in body:
                continue
            assert 'Records' in body, 'Unexpected event received'
            doc = json.loads(body)
            events.extend(doc['Records'])
        assert len(events) >= min_events
    retry(collect_events, retries=timeout, sleep=0.01)
    return events

@pytest.fixture
def sqs_create_queue_with_client():
    if False:
        i = 10
        return i + 15
    queue_urls = []

    def factory(sqs_client, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'QueueName' not in kwargs:
            kwargs['QueueName'] = 'test-queue-%s' % short_uid()
        response = sqs_client.create_queue(**kwargs)
        url = response['QueueUrl']
        queue_urls.append((sqs_client, url))
        return url
    yield factory
    for (client, queue_url) in queue_urls:
        try:
            client.delete_queue(QueueUrl=queue_url)
        except Exception as e:
            LOG.debug('error cleaning up queue %s: %s', queue_url, e)

class TestS3NotificationsToSQS:

    @markers.aws.validated
    def test_object_created_put(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            while True:
                i = 10
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectCreated:Put'])
        aws_client.s3.put_bucket_versioning(Bucket=bucket_name, VersioningConfiguration={'Status': 'Enabled'})
        obj0 = aws_client.s3.put_object(Bucket=bucket_name, Key='my_key_0', Body='something')
        obj1 = aws_client.s3.put_object(Bucket=bucket_name, Key='my_key_1', Body='something else')
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, min_events=2)
        assert len(events) == 2, f'unexpected number of events in {events}'
        events.sort(key=lambda x: x['s3']['object']['size'])
        snapshot.match('receive_messages', {'messages': events})
        assert events[0]['eventSource'] == 'aws:s3'
        assert events[0]['eventName'] == 'ObjectCreated:Put'
        assert events[0]['s3']['bucket']['name'] == bucket_name
        assert events[0]['s3']['object']['key'] == 'my_key_0'
        assert events[0]['s3']['object']['size'] == 9
        assert events[0]['s3']['object']['versionId']
        assert obj0['VersionId'] == events[0]['s3']['object']['versionId']
        assert events[1]['eventSource'] == 'aws:s3'
        assert events[0]['eventName'] == 'ObjectCreated:Put'
        assert events[1]['s3']['bucket']['name'] == bucket_name
        assert events[1]['s3']['object']['key'] == 'my_key_1'
        assert events[1]['s3']['object']['size'] == 14
        assert events[1]['s3']['object']['versionId']
        assert obj1['VersionId'] == events[1]['s3']['object']['versionId']

    @markers.aws.validated
    def test_object_created_copy(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            return 10
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        snapshot.add_transformer(snapshot.transform.jsonpath('$..s3.object.key', 'object-key'))
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectCreated:Copy'])
        src_key = 'src-dest-%s' % short_uid()
        dest_key = 'key-dest-%s' % short_uid()
        aws_client.s3.put_object(Bucket=bucket_name, Key=src_key, Body='something')
        assert not sqs_collect_s3_events(aws_client.sqs, queue_url, 0, timeout=1), 'unexpected event triggered for put_object'
        aws_client.s3.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': src_key}, Key=dest_key)
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, 1)
        assert len(events) == 1, f'unexpected number of events in {events}'
        snapshot.match('receive_messages', {'messages': events})
        assert events[0]['eventSource'] == 'aws:s3'
        assert events[0]['eventName'] == 'ObjectCreated:Copy'
        assert events[0]['s3']['bucket']['name'] == bucket_name
        assert events[0]['s3']['object']['key'] == dest_key

    @markers.aws.validated
    def test_object_created_and_object_removed(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            for i in range(10):
                print('nop')
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        snapshot.add_transformer(snapshot.transform.jsonpath('$..s3.object.key', 'object-key'))
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectCreated:*', 's3:ObjectRemoved:*'])
        src_key = 'src-dest-%s' % short_uid()
        dest_key = 'key-dest-%s' % short_uid()
        aws_client.s3.put_object(Bucket=bucket_name, Key=src_key, Body='something')
        aws_client.s3.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': src_key}, Key=dest_key)
        aws_client.s3.delete_object(Bucket=bucket_name, Key=src_key)
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, 3)
        assert len(events) == 3, f'unexpected number of events in {events}'
        events.sort(key=lambda x: x['eventName'])
        snapshot.match('receive_messages', {'messages': events})
        assert events[1]['eventName'] == 'ObjectCreated:Put'
        assert events[1]['s3']['bucket']['name'] == bucket_name
        assert events[1]['s3']['object']['key'] == src_key
        assert events[0]['eventName'] == 'ObjectCreated:Copy'
        assert events[0]['s3']['bucket']['name'] == bucket_name
        assert events[0]['s3']['object']['key'] == dest_key
        assert events[2]['eventName'] == 'ObjectRemoved:Delete'
        assert events[2]['s3']['bucket']['name'] == bucket_name
        assert events[2]['s3']['object']['key'] == src_key

    @markers.aws.validated
    def test_delete_objects(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            while True:
                i = 10
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        snapshot.add_transformer(snapshot.transform.jsonpath('$..s3.object.key', 'object-key'))
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectRemoved:*'])
        key = 'key-%s' % short_uid()
        aws_client.s3.put_object(Bucket=bucket_name, Key=key, Body='something')
        aws_client.s3.delete_objects(Bucket=bucket_name, Delete={'Objects': [{'Key': key}, {'Key': 'dummy1'}, {'Key': 'dummy2'}], 'Quiet': True})
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, 3)
        assert len(events) == 3, f'unexpected number of events in {events}'
        events.sort(key=lambda x: x['s3']['object']['key'])
        snapshot.match('receive_messages', {'messages': events})
        assert events[2]['eventName'] == 'ObjectRemoved:Delete'
        assert events[2]['s3']['bucket']['name'] == bucket_name
        assert events[2]['s3']['object']['key'] == key

    @markers.aws.validated
    def test_object_created_complete_multipart_upload(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, tmpdir, snapshot, aws_client):
        if False:
            return 10
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        key = 'test-key'
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectCreated:*'])
        config = TransferConfig(multipart_threshold=5 * KB, multipart_chunksize=1 * KB)
        file = tmpdir / 'test-file.bin'
        data = b'1' * (6 * KB)
        file.write(data=data, mode='w')
        aws_client.s3.upload_file(Bucket=bucket_name, Key=key, Filename=str(file.realpath()), Config=config)
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, 1)
        snapshot.match('receive_messages', {'messages': events})
        assert events[0]['eventName'] == 'ObjectCreated:CompleteMultipartUpload'
        assert events[0]['s3']['bucket']['name'] == bucket_name
        assert events[0]['s3']['object']['key'] == key
        assert events[0]['s3']['object']['size'] == file.size()

    @markers.aws.validated
    def test_key_encoding(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            for i in range(10):
                print('nop')
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectCreated:*'])
        key = 'a@b'
        key_encoded = 'a%40b'
        aws_client.s3.put_object(Bucket=bucket_name, Key=key, Body='something')
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, min_events=1)
        snapshot.match('receive_messages', {'messages': events})
        assert events[0]['eventName'] == 'ObjectCreated:Put'
        assert events[0]['s3']['object']['key'] == key_encoded

    @markers.aws.validated
    def test_object_created_put_with_presigned_url_upload(self, s3_create_bucket, sqs_create_queue, sqs_create_queue_with_client, s3_create_sqs_bucket_notification, snapshot, aws_client, aws_client_factory):
        if False:
            return 10
        "This test validates that pre-signed URL works with notification, and that the awsRegion field is the\n        bucket's region"
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        snapshot.add_transformer(snapshot.transform.key_value('awsRegion'), priority=-1)
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        key = 'key-by-hostname'
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectCreated:*'])
        url = aws_client.s3.generate_presigned_url('put_object', Params={'Bucket': bucket_name, 'Key': key})
        requests.put(url, data='something', verify=False)
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, 1)
        snapshot.match('receive_messages', {'messages': events})
        assert events[0]['eventName'] == 'ObjectCreated:Put'
        assert events[0]['s3']['object']['key'] == key
        bucket_name_region_2 = s3_create_bucket(CreateBucketConfiguration={'LocationConstraint': SECONDARY_TEST_AWS_REGION_NAME})
        sqs_client_region_2 = aws_client_factory(region_name=SECONDARY_TEST_AWS_REGION_NAME).sqs
        queue_url_region_2 = sqs_create_queue_with_client(sqs_client_region_2)
        s3_create_sqs_bucket_notification(bucket_name=bucket_name_region_2, queue_url=queue_url_region_2, events=['s3:ObjectCreated:*'], sqs_client=sqs_client_region_2)
        url_bucket_region_2 = aws_client.s3.generate_presigned_url('put_object', Params={'Bucket': bucket_name_region_2, 'Key': key})
        requests.put(url_bucket_region_2, data='something', verify=False)
        events = sqs_collect_s3_events(sqs_client_region_2, queue_url_region_2, 1)
        snapshot.match('receive_messages_region_2', {'messages': events})

    @markers.aws.validated
    def test_object_tagging_put_event(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            print('Hello World!')
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        snapshot.add_transformer(snapshot.transform.jsonpath('$..s3.object.key', 'object-key'))
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectTagging:Put'])
        dest_key = 'key-dest-%s' % short_uid()
        aws_client.s3.put_object(Bucket=bucket_name, Key=dest_key, Body='FooBarBlitz')
        assert not sqs_collect_s3_events(aws_client.sqs, queue_url, 0, timeout=1), 'unexpected event triggered for put_object'
        aws_client.s3.put_object_tagging(Bucket=bucket_name, Key=dest_key, Tagging={'TagSet': [{'Key': 'swallow_type', 'Value': 'african'}]})
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, 1)
        assert len(events) == 1, f'unexpected number of events in {events}'
        snapshot.match('receive_messages', {'messages': events})
        assert events[0]['eventSource'] == 'aws:s3'
        assert events[0]['eventName'] == 'ObjectTagging:Put'
        assert events[0]['s3']['bucket']['name'] == bucket_name
        assert events[0]['s3']['object']['key'] == dest_key

    @markers.aws.validated
    def test_object_tagging_delete_event(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            return 10
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        snapshot.add_transformer(snapshot.transform.jsonpath('$..s3.object.key', 'object-key'))
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectTagging:Delete'])
        dest_key = 'key-dest-%s' % short_uid()
        aws_client.s3.put_object(Bucket=bucket_name, Key=dest_key, Body='FooBarBlitz')
        assert not sqs_collect_s3_events(aws_client.sqs, queue_url, 0, timeout=1), 'unexpected event triggered for put_object'
        aws_client.s3.put_object_tagging(Bucket=bucket_name, Key=dest_key, Tagging={'TagSet': [{'Key': 'swallow_type', 'Value': 'african'}]})
        aws_client.s3.delete_object_tagging(Bucket=bucket_name, Key=dest_key)
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, 1)
        assert len(events) == 1, f'unexpected number of events in {events}'
        snapshot.match('receive_messages', {'messages': events})
        assert events[0]['eventSource'] == 'aws:s3'
        assert events[0]['eventName'] == 'ObjectTagging:Delete'
        assert events[0]['s3']['bucket']['name'] == bucket_name
        assert events[0]['s3']['object']['key'] == dest_key

    @markers.aws.validated
    def test_xray_header(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, cleanups, snapshot, aws_client):
        if False:
            while True:
                i = 10
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        snapshot.add_transformer(snapshot.transform.key_value('MD5OfBody', reference_replacement=False))

        def add_xray_header(request, **kwargs):
            if False:
                return 10
            request.headers['X-Amzn-Trace-Id'] = 'Root=1-3152b799-8954dae64eda91bc9a23a7e8;Parent=7fa8c0f79203be72;Sampled=1'
        aws_client.s3.meta.events.register('before-send.s3.*', add_xray_header)
        cleanups.append(lambda : aws_client.s3.meta.events.unregister('before-send.s3.*', add_xray_header))
        key = 'test-data'
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectCreated:*'])
        aws_client.s3.put_object(Bucket=bucket_name, Key=key, Body='something')

        def get_messages():
            if False:
                while True:
                    i = 10
            recv_messages = []
            resp = aws_client.sqs.receive_message(QueueUrl=queue_url, AttributeNames=['AWSTraceHeader'], MessageAttributeNames=['All'], VisibilityTimeout=0)
            for m in resp['Messages']:
                if 's3:TestEvent' in m['Body']:
                    aws_client.sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=m['ReceiptHandle'])
                    continue
                recv_messages.append(m)
            assert len(recv_messages) >= 1
            return recv_messages
        messages = retry(get_messages, retries=10)
        assert 'AWSTraceHeader' in messages[0]['Attributes']
        assert messages[0]['Attributes']['AWSTraceHeader'] == 'Root=1-3152b799-8954dae64eda91bc9a23a7e8;Parent=7fa8c0f79203be72;Sampled=1'
        snapshot.match('receive_messages', {'messages': messages})

    @markers.aws.validated
    def test_notifications_with_filter(self, s3_create_bucket, s3_create_sqs_bucket_notification, sqs_create_queue, snapshot, aws_client):
        if False:
            for i in range(10):
                print('nop')
        bucket_name = f'notification-bucket-{short_uid()}'
        s3_create_bucket(Bucket=bucket_name)
        queue_name = f'queue-{short_uid()}'
        queue_url = sqs_create_queue(QueueName=queue_name)
        snapshot.add_transformer(snapshot.transform.regex(queue_name, '<queue>'))
        snapshot.add_transformer(snapshot.transform.regex(bucket_name, '<bucket>'))
        snapshot.add_transformer(snapshot.transform.s3_notifications_transformer())
        queue_arn = set_policy_for_queue(aws_client.sqs, queue_url, bucket_name)
        events = ['s3:ObjectCreated:*', 's3:ObjectRemoved:Delete']
        filter_rules = {'FilterRules': [{'Name': 'Prefix', 'Value': 'testupload/'}, {'Name': 'Suffix', 'Value': 'testfile.txt'}]}
        aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration={'QueueConfigurations': [{'Id': 'id0001', 'QueueArn': queue_arn, 'Events': events, 'Filter': {'Key': filter_rules}}, {'Id': 'id0002', 'QueueArn': queue_arn, 'Events': ['s3:ObjectTagging:*'], 'Filter': {'Key': filter_rules}}]})
        config = aws_client.s3.get_bucket_notification_configuration(Bucket=bucket_name)
        snapshot.match('config', config)
        assert 2 == len(config['QueueConfigurations'])
        config = [c for c in config['QueueConfigurations'] if c.get('Events')][0]
        assert events == config['Events']
        assert filter_rules == config['Filter']['Key']
        test_key1 = '/testdata'
        test_data1 = b'{"test": "bucket_notification1"}'
        aws_client.s3.upload_fileobj(BytesIO(test_data1), bucket_name, test_key1)
        test_key2 = 'testupload/dir1/testfile.txt'
        test_data2 = b'{"test": "bucket_notification2"}'
        aws_client.s3.upload_fileobj(BytesIO(test_data2), bucket_name, test_key2)
        messages = sqs_collect_s3_events(aws_client.sqs, queue_url, 1)
        assert len(messages) == 1
        snapshot.match('message', messages[0])
        assert messages[0]['s3']['object']['key'] == test_key2
        assert messages[0]['s3']['bucket']['name'] == bucket_name
        aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration={})
        config = aws_client.s3.get_bucket_notification_configuration(Bucket=bucket_name)
        snapshot.match('config_empty', config)
        assert not config.get('QueueConfigurations')
        assert not config.get('TopicConfiguration')
        event = 's3:ObjectCreated:*'
        aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration={'QueueConfigurations': [{'Id': 'id123456', 'QueueArn': queue_arn, 'Events': [event]}]})
        config = aws_client.s3.get_bucket_notification_configuration(Bucket=bucket_name)
        snapshot.match('config_updated', config)
        config = config['QueueConfigurations'][0]
        assert [event] == config['Events']
        event = 's3:ObjectCreated:*'
        filter_rules = {'FilterRules': [{'Name': 'Prefix', 'Value': 'testupload/'}]}
        aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration={'QueueConfigurations': [{'Id': 'id123456', 'QueueArn': queue_arn, 'Events': [event], 'Filter': {'Key': filter_rules}}]})
        config = aws_client.s3.get_bucket_notification_configuration(Bucket=bucket_name)
        snapshot.match('config_updated_filter', config)
        config = config['QueueConfigurations'][0]
        assert [event] == config['Events']
        assert filter_rules == config['Filter']['Key']

    @markers.aws.validated
    def test_filter_rules_case_insensitive(self, s3_create_bucket, sqs_create_queue, snapshot, aws_client):
        if False:
            print('Hello World!')
        bucket_name = s3_create_bucket()
        id = short_uid()
        queue_url = sqs_create_queue(QueueName=f'my-queue-{id}')
        queue_attributes = aws_client.sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['QueueArn'])
        snapshot.add_transformer(snapshot.transform.key_value('Id', 'id'))
        snapshot.add_transformer(snapshot.transform.regex(id, '<queue_id>'))
        cfg = {'QueueConfigurations': [{'QueueArn': queue_attributes['Attributes']['QueueArn'], 'Events': ['s3:ObjectCreated:*'], 'Filter': {'Key': {'FilterRules': [{'Name': 'suffix', 'Value': '.txt'}, {'Name': 'PREFIX', 'Value': 'notif-'}]}}}]}
        aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=cfg, SkipDestinationValidation=True)
        response = aws_client.s3.get_bucket_notification_configuration(Bucket=bucket_name)
        rules = response['QueueConfigurations'][0]['Filter']['Key']['FilterRules']
        valid = ['Prefix', 'Suffix']
        response['QueueConfigurations'][0]['Filter']['Key']['FilterRules'].sort(key=lambda x: x['Name'])
        assert rules[0]['Name'] in valid
        assert rules[1]['Name'] in valid
        snapshot.match('bucket_notification_configuration', response)

    @markers.snapshot.skip_snapshot_verify(paths=['$..Error.ArgumentName', '$..Error.ArgumentValue'])
    @markers.aws.validated
    def test_bucket_notification_with_invalid_filter_rules(self, s3_create_bucket, sqs_create_queue, snapshot, aws_client):
        if False:
            while True:
                i = 10
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        queue_attributes = aws_client.sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['QueueArn'])
        cfg = {'QueueConfigurations': [{'QueueArn': queue_attributes['Attributes']['QueueArn'], 'Events': ['s3:ObjectCreated:*'], 'Filter': {'Key': {'FilterRules': [{'Name': 'INVALID', 'Value': 'does not matter'}]}}}]}
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=cfg)
        snapshot.match('invalid_filter_name', e.value.response)

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..Error.ArgumentName1', '$..Error.ArgumentValue1', '$..Error.ArgumentName', '$..Error.ArgumentValue'])
    def test_invalid_sqs_arn(self, s3_create_bucket, account_id, snapshot, aws_client):
        if False:
            print('Hello World!')
        bucket_name = s3_create_bucket()
        config = {'QueueConfigurations': [{'Id': 'id123', 'Events': ['s3:ObjectCreated:*']}]}
        config['QueueConfigurations'][0]['QueueArn'] = 'invalid-queue'
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=config, SkipDestinationValidation=False)
        snapshot.match('invalid_not_skip', e.value.response)
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=config, SkipDestinationValidation=True)
        snapshot.match('invalid_skip', e.value.response)
        config['QueueConfigurations'][0]['QueueArn'] = arns.sqs_queue_arn('my-queue', account_id=account_id, region_name=aws_client.s3.meta.region_name)
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=config)
        snapshot.match('queue-does-not-exist', e.value.response)
        aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=config, SkipDestinationValidation=True)
        config = aws_client.s3.get_bucket_notification_configuration(Bucket=bucket_name)
        snapshot.match('skip_destination_validation', config)

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..Error.ArgumentName', '$..Error.ArgumentValue', '$..Error.ArgumentName1', '$..Error.ArgumentValue1', '$..Error.ArgumentName2', '$..Error.ArgumentValue2', '$..Error.Message'])
    def test_multiple_invalid_sqs_arns(self, s3_create_bucket, account_id, snapshot, aws_client):
        if False:
            print('Hello World!')
        bucket_name = s3_create_bucket()
        config = {'QueueConfigurations': [{'Id': 'id1', 'Events': ['s3:ObjectCreated:*'], 'QueueArn': 'invalid_arn'}, {'Id': 'id2', 'Events': ['s3:ObjectRemoved:*'], 'QueueArn': 'invalid_arn_2'}]}
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=config)
        snapshot.match('two-queue-arns-invalid', e.value.response)
        config['QueueConfigurations'][0]['QueueArn'] = arns.sqs_queue_arn('my-queue', account_id=account_id, region_name=aws_client.s3.meta.region_name)
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=config)
        snapshot.match('one-queue-invalid-one-not-existent', e.value.response)
        config['QueueConfigurations'][1]['QueueArn'] = arns.sqs_queue_arn('my-queue-2', account_id=account_id, region_name=aws_client.s3.meta.region_name)
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_notification_configuration(Bucket=bucket_name, NotificationConfiguration=config)
        snapshot.match('multiple-queues-do-not-exist', e.value.response)

    @markers.aws.validated
    def test_object_put_acl(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            return 10
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        bucket_name = s3_create_bucket()
        aws_client.s3.delete_bucket_ownership_controls(Bucket=bucket_name)
        aws_client.s3.delete_public_access_block(Bucket=bucket_name)
        queue_url = sqs_create_queue()
        key_name = 'my_key_acl'
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectAcl:Put'])
        aws_client.s3.put_object(Bucket=bucket_name, Key=key_name, Body='something')
        list_bucket_output = aws_client.s3.list_buckets()
        owner = list_bucket_output['Owner']
        aws_client.s3.put_object_acl(Bucket=bucket_name, Key=key_name, ACL='private')
        aws_client.s3.put_object_acl(Bucket=bucket_name, Key=key_name, ACL='public-read')
        aws_client.s3.put_object_acl(Bucket=bucket_name, Key=key_name, GrantRead='uri="http://acs.amazonaws.com/groups/s3/LogDelivery"')
        acp = {'Owner': owner, 'Grants': [{'Grantee': {'ID': owner['ID'], 'Type': 'CanonicalUser'}, 'Permission': 'FULL_CONTROL'}, {'Grantee': {'URI': 'http://acs.amazonaws.com/groups/s3/LogDelivery', 'Type': 'Group'}, 'Permission': 'WRITE'}]}
        aws_client.s3.put_object_acl(Bucket=bucket_name, Key=key_name, AccessControlPolicy=acp)
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, min_events=3)
        assert len(events) == 3, f'unexpected number of events in {events}'
        events.sort(key=lambda x: x['eventTime'])
        snapshot.match('receive_messages', {'messages': events})

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..messages[1].requestParameters.sourceIPAddress'])
    def test_restore_object(self, s3_create_bucket, sqs_create_queue, s3_create_sqs_bucket_notification, snapshot, aws_client):
        if False:
            while True:
                i = 10
        snapshot.add_transformer(snapshot.transform.sqs_api())
        snapshot.add_transformer(snapshot.transform.s3_api())
        bucket_name = s3_create_bucket()
        queue_url = sqs_create_queue()
        key_name = 'my_key_restore'
        s3_create_sqs_bucket_notification(bucket_name, queue_url, ['s3:ObjectRestore:*'])
        aws_client.s3.put_object(Bucket=bucket_name, Key=key_name, Body='something', StorageClass='GLACIER')
        aws_client.s3.restore_object(Bucket=bucket_name, Key=key_name, RestoreRequest={'Days': 1, 'GlacierJobParameters': {'Tier': 'Expedited'}})

        def _is_object_restored():
            if False:
                print('Hello World!')
            resp = aws_client.s3.head_object(Bucket=bucket_name, Key=key_name)
            assert 'ongoing-request="false"' in resp['Restore']
        if is_aws_cloud():
            retries = 12
            sleep = 30
        else:
            retries = 3
            sleep = 1
        retry(_is_object_restored, retries=retries, sleep=sleep)
        events = sqs_collect_s3_events(aws_client.sqs, queue_url, min_events=2)
        assert len(events) == 2, f'unexpected number of events in {events}'
        events.sort(key=lambda x: x['eventTime'])
        snapshot.match('receive_messages', {'messages': events})