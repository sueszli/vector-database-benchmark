import json
import boto3
import pytest
from dagster import job, op
from dagster_aws.cloudwatch import cloudwatch_logger
from moto import mock_logs

@op
def hello_op(context):
    if False:
        for i in range(10):
            print('nop')
    context.log.info('Hello, Cloudwatch!')
    context.log.error('This is an error')

@job(logger_defs={'cloudwatch': cloudwatch_logger})
def hello_cloudwatch():
    if False:
        print('Hello World!')
    hello_op()

@pytest.fixture
def region():
    if False:
        i = 10
        return i + 15
    return 'us-east-1'

@pytest.fixture
def cloudwatch_client(region):
    if False:
        i = 10
        return i + 15
    with mock_logs():
        yield boto3.client('logs', region_name=region)

@pytest.fixture
def log_group(cloudwatch_client):
    if False:
        while True:
            i = 10
    name = '/dagster-test/test-cloudwatch-logging'
    cloudwatch_client.create_log_group(logGroupName=name)
    return name

@pytest.fixture
def log_stream(cloudwatch_client, log_group):
    if False:
        for i in range(10):
            print('nop')
    name = 'test-logging'
    cloudwatch_client.create_log_stream(logGroupName=log_group, logStreamName=name)
    return name

def test_cloudwatch_logging_bad_log_group_name(region, log_stream):
    if False:
        while True:
            i = 10
    with pytest.raises(Exception, match='Failed to initialize Cloudwatch logger: Could not find log group with name fake-log-group'):
        hello_cloudwatch.execute_in_process({'loggers': {'cloudwatch': {'config': {'log_group_name': 'fake-log-group', 'log_stream_name': log_stream, 'aws_region': region}}}})

def test_cloudwatch_logging_bad_log_stream_name(region, log_group):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(Exception, match='Failed to initialize Cloudwatch logger: Could not find log stream with name fake-log-stream'):
        hello_cloudwatch.execute_in_process({'loggers': {'cloudwatch': {'config': {'log_group_name': log_group, 'log_stream_name': 'fake-log-stream', 'aws_region': region}}}})

def test_cloudwatch_logging_bad_region(log_group, log_stream):
    if False:
        return 10
    with pytest.raises(Exception, match=f'Failed to initialize Cloudwatch logger: Could not find log group with name {log_group}'):
        hello_cloudwatch.execute_in_process({'loggers': {'cloudwatch': {'config': {'log_group_name': log_group, 'log_stream_name': log_stream, 'aws_region': 'us-west-1'}}}})

def test_cloudwatch_logging(region, cloudwatch_client, log_group, log_stream):
    if False:
        print('Hello World!')
    hello_cloudwatch.execute_in_process({'loggers': {'cloudwatch': {'config': {'log_group_name': log_group, 'log_stream_name': log_stream, 'aws_region': region}}}})
    events = cloudwatch_client.get_log_events(logGroupName=log_group, logStreamName=log_stream)['events']
    info_message = json.loads(events[0]['message'])
    error_message = json.loads(events[1]['message'])
    assert info_message['levelname'] == 'INFO'
    assert info_message['dagster_meta']['orig_message'] == 'Hello, Cloudwatch!'
    assert error_message['levelname'] == 'ERROR'
    assert error_message['dagster_meta']['orig_message'] == 'This is an error'