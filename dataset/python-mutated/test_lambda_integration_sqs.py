import json
import os
import time
import pytest
from botocore.exceptions import ClientError
from localstack.aws.api.lambda_ import InvalidParameterValueException, Runtime
from localstack.testing.aws.lambda_utils import _await_event_source_mapping_enabled
from localstack.testing.aws.util import is_aws_cloud
from localstack.testing.pytest import markers
from localstack.utils.strings import short_uid
from localstack.utils.sync import retry
from localstack.utils.testutil import check_expected_lambda_log_events_length, get_lambda_log_events
from tests.aws.services.lambda_.functions import lambda_integration
from tests.aws.services.lambda_.test_lambda import TEST_LAMBDA_PYTHON, TEST_LAMBDA_PYTHON_ECHO, TEST_LAMBDA_PYTHON_ECHO_VERSION_ENV
THIS_FOLDER = os.path.dirname(os.path.realpath(__file__))
LAMBDA_SQS_INTEGRATION_FILE = os.path.join(THIS_FOLDER, 'functions', 'lambda_sqs_integration.py')
LAMBDA_SQS_BATCH_ITEM_FAILURE_FILE = os.path.join(THIS_FOLDER, 'functions/lambda_sqs_batch_item_failure.py')
DEFAULT_SQS_BATCH_SIZE = 10
MAX_SQS_BATCH_SIZE_FIFO = 10

def _await_queue_size(sqs_client, queue_url: str, qsize: int, retries=10, sleep=1):
    if False:
        print('Hello World!')

    def _verify_event_queue_size():
        if False:
            return 10
        attr = 'ApproximateNumberOfMessages'
        _approx = int(sqs_client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=[attr])['Attributes'][attr])
        assert _approx >= qsize
    retry(_verify_event_queue_size, retries=retries, sleep=sleep)

@pytest.fixture(autouse=True)
def _snapshot_transformers(snapshot):
    if False:
        i = 10
        return i + 15
    snapshot.add_transformer(snapshot.transform.key_value('QueueUrl'))
    snapshot.add_transformer(snapshot.transform.key_value('ReceiptHandle'))
    snapshot.add_transformer(snapshot.transform.key_value('SenderId', reference_replacement=False))
    snapshot.add_transformer(snapshot.transform.key_value('SequenceNumber'))
    snapshot.add_transformer(snapshot.transform.resource_name())
    snapshot.add_transformer(snapshot.transform.key_value('MD5OfBody'))
    snapshot.add_transformer(snapshot.transform.key_value('receiptHandle'))
    snapshot.add_transformer(snapshot.transform.key_value('md5OfBody'))

@markers.snapshot.skip_snapshot_verify(paths=['$..ParallelizationFactor', '$..LastProcessingResult', '$..Topics', '$..MaximumRetryAttempts', '$..MaximumBatchingWindowInSeconds', '$..FunctionResponseTypes', '$..StartingPosition', '$..StateTransitionReason'])
@markers.aws.validated
def test_failing_lambda_retries_after_visibility_timeout(create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
    if False:
        while True:
            i = 10
    'This test verifies a basic SQS retry scenario. The lambda uses an SQS queue as event source, and we are\n    testing whether the lambda automatically retries after the visibility timeout expires, and, after the retry,\n    properly deletes the message from the queue.'
    destination_queue_name = f'destination-queue-{short_uid()}'
    destination_url = sqs_create_queue(QueueName=destination_queue_name)
    snapshot.match('get_destination_queue_url', aws_client.sqs.get_queue_url(QueueName=destination_queue_name))
    retry_timeout = 5
    function_name = f'failing-lambda-{short_uid()}'
    create_lambda_function(func_name=function_name, handler_file=LAMBDA_SQS_INTEGRATION_FILE, runtime=Runtime.python3_8, role=lambda_su_role, timeout=retry_timeout)
    event_source_url = sqs_create_queue(QueueName=f'source-queue-{short_uid()}', Attributes={'VisibilityTimeout': str(retry_timeout)})
    event_source_arn = sqs_get_queue_arn(event_source_url)
    response = aws_client.lambda_.create_event_source_mapping(EventSourceArn=event_source_arn, FunctionName=function_name, BatchSize=1)
    mapping_uuid = response['UUID']
    cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
    _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
    response = aws_client.lambda_.get_event_source_mapping(UUID=mapping_uuid)
    snapshot.match('event_source_mapping', response)
    event = {'destination': destination_url, 'fail_attempts': 1}
    aws_client.sqs.send_message(QueueUrl=event_source_url, MessageBody=json.dumps(event))
    then = time.time()
    first_response = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=15, MaxNumberOfMessages=1)
    assert 'Messages' in first_response
    snapshot.match('first_attempt', first_response)
    second_response = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=15, MaxNumberOfMessages=1)
    assert 'Messages' in second_response
    snapshot.match('second_attempt', second_response)
    assert time.time() >= then + retry_timeout
    third_response = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=retry_timeout + 1, MaxNumberOfMessages=1)
    assert 'Messages' in third_response
    assert third_response['Messages'] == []

@markers.snapshot.skip_snapshot_verify(paths=['$..stringListValues', '$..binaryListValues'])
@markers.aws.validated
def test_message_body_and_attributes_passed_correctly(create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
    if False:
        i = 10
        return i + 15
    destination_queue_name = f'destination-queue-{short_uid()}'
    destination_url = sqs_create_queue(QueueName=destination_queue_name)
    snapshot.match('get_destination_queue_url', aws_client.sqs.get_queue_url(QueueName=destination_queue_name))
    retry_timeout = 5
    retries = 2
    function_name = f'lambda-{short_uid()}'
    create_lambda_function(func_name=function_name, handler_file=LAMBDA_SQS_INTEGRATION_FILE, runtime=Runtime.python3_8, role=lambda_su_role, timeout=retry_timeout)
    event_dlq_url = sqs_create_queue(QueueName=f'event-dlq-{short_uid()}')
    event_dlq_arn = sqs_get_queue_arn(event_dlq_url)
    event_source_url = sqs_create_queue(QueueName=f'source-queue-{short_uid()}', Attributes={'VisibilityTimeout': str(retry_timeout), 'RedrivePolicy': json.dumps({'deadLetterTargetArn': event_dlq_arn, 'maxReceiveCount': retries})})
    event_source_arn = sqs_get_queue_arn(event_source_url)
    mapping_uuid = aws_client.lambda_.create_event_source_mapping(EventSourceArn=event_source_arn, FunctionName=function_name, BatchSize=1)['UUID']
    cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
    _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
    event = {'destination': destination_url, 'fail_attempts': 0}
    aws_client.sqs.send_message(QueueUrl=event_source_url, MessageBody=json.dumps(event), MessageAttributes={'Title': {'DataType': 'String', 'StringValue': 'The Whistler'}, 'Author': {'DataType': 'String', 'StringValue': 'John Grisham'}, 'WeeksOn': {'DataType': 'Number', 'StringValue': '6'}})
    response = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=15, MaxNumberOfMessages=1)
    assert 'Messages' in response
    snapshot.match('first_attempt', response)

@markers.snapshot.skip_snapshot_verify(paths=['$..ParallelizationFactor', '$..LastProcessingResult', '$..Topics', '$..MaximumRetryAttempts', '$..MaximumBatchingWindowInSeconds', '$..FunctionResponseTypes', '$..StartingPosition', '$..StateTransitionReason'])
@markers.aws.validated
def test_redrive_policy_with_failing_lambda(create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
    if False:
        i = 10
        return i + 15
    'This test verifies that SQS moves a message that is passed to a failing lambda to a DLQ according to the\n    redrive policy, and the lambda is invoked the correct number of times. The test retries twice and the event\n    source mapping should then automatically move the message to the DLQ, but not earlier (see\n    https://github.com/localstack/localstack/issues/5283)'
    destination_queue_name = f'destination-queue-{short_uid()}'
    destination_url = sqs_create_queue(QueueName=destination_queue_name)
    snapshot.match('get_destination_queue_url', aws_client.sqs.get_queue_url(QueueName=destination_queue_name))
    retry_timeout = 5
    retries = 2
    function_name = f'failing-lambda-{short_uid()}'
    create_lambda_function(func_name=function_name, handler_file=LAMBDA_SQS_INTEGRATION_FILE, runtime=Runtime.python3_8, role=lambda_su_role, timeout=retry_timeout)
    event_dlq_url = sqs_create_queue(QueueName=f'event-dlq-{short_uid()}')
    event_dlq_arn = sqs_get_queue_arn(event_dlq_url)
    event_source_url = sqs_create_queue(QueueName=f'source-queue-{short_uid()}', Attributes={'VisibilityTimeout': str(retry_timeout), 'RedrivePolicy': json.dumps({'deadLetterTargetArn': event_dlq_arn, 'maxReceiveCount': retries})})
    event_source_arn = sqs_get_queue_arn(event_source_url)
    mapping_uuid = aws_client.lambda_.create_event_source_mapping(EventSourceArn=event_source_arn, FunctionName=function_name, BatchSize=1)['UUID']
    cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
    _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
    event = {'destination': destination_url, 'fail_attempts': retries}
    aws_client.sqs.send_message(QueueUrl=event_source_url, MessageBody=json.dumps(event))
    first_response = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=15, MaxNumberOfMessages=1)
    assert 'Messages' in first_response
    snapshot.match('first_attempt', first_response)
    second_response = aws_client.sqs.receive_message(QueueUrl=event_dlq_url, WaitTimeSeconds=1)
    assert 'Messages' in second_response
    assert second_response['Messages'] == []
    third_response = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=15, MaxNumberOfMessages=1)
    assert 'Messages' in third_response
    snapshot.match('second_attempt', third_response)
    dlq_response = aws_client.sqs.receive_message(QueueUrl=event_dlq_url, WaitTimeSeconds=15)
    assert 'Messages' in dlq_response
    snapshot.match('dlq_response', dlq_response)

@markers.aws.validated
def test_sqs_queue_as_lambda_dead_letter_queue(lambda_su_role, create_lambda_function, sqs_create_queue, sqs_get_queue_arn, snapshot, aws_client):
    if False:
        for i in range(10):
            print('nop')
    snapshot.add_transformer([snapshot.transform.key_value('MD5OfMessageAttributes', value_replacement='<md5-hash>', reference_replacement=False), snapshot.transform.jsonpath('$..Messages..MessageAttributes.RequestID.StringValue', 'request-id')])
    dlq_queue_url = sqs_create_queue()
    dlq_queue_arn = sqs_get_queue_arn(dlq_queue_url)
    function_name = f'lambda-fn-{short_uid()}'
    lambda_creation_response = create_lambda_function(func_name=function_name, handler_file=TEST_LAMBDA_PYTHON, runtime=Runtime.python3_9, role=lambda_su_role, DeadLetterConfig={'TargetArn': dlq_queue_arn})
    snapshot.match('lambda-response-dlq-config', lambda_creation_response['CreateFunctionResponse']['DeadLetterConfig'])
    aws_client.lambda_.put_function_event_invoke_config(FunctionName=function_name, MaximumRetryAttempts=0)
    payload = {lambda_integration.MSG_BODY_RAISE_ERROR_FLAG: 1}
    aws_client.lambda_.invoke(FunctionName=function_name, Payload=json.dumps(payload), InvocationType='Event')

    def receive_dlq():
        if False:
            return 10
        result = aws_client.sqs.receive_message(QueueUrl=dlq_queue_url, MessageAttributeNames=['All'], VisibilityTimeout=0)
        assert len(result['Messages']) > 0
        return result
    sleep = 3 if is_aws_cloud() else 1
    messages = retry(receive_dlq, retries=30, sleep=sleep)
    snapshot.match('messages', messages)

@markers.snapshot.skip_snapshot_verify(paths=['$..SequenceNumber', '$..receiptHandle', '$..md5OfBody', '$..MD5OfMessageBody', '$..create_event_source_mapping.ParallelizationFactor', '$..create_event_source_mapping.LastProcessingResult', '$..create_event_source_mapping.Topics', '$..create_event_source_mapping.MaximumRetryAttempts', '$..create_event_source_mapping.MaximumBatchingWindowInSeconds', '$..create_event_source_mapping.FunctionResponseTypes', '$..create_event_source_mapping.StartingPosition', '$..create_event_source_mapping.StateTransitionReason', '$..create_event_source_mapping.State', '$..create_event_source_mapping.ResponseMetadata'])
@markers.aws.validated
def test_report_batch_item_failures(create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
    if False:
        print('Hello World!')
    'This test verifies the SQS Lambda integration feature Reporting batch item failures\n    redrive policy, and the lambda is invoked the correct number of times. The test retries twice and the event\n    source mapping should then automatically move the message to the DLQ, but not earlier (see\n    https://github.com/localstack/localstack/issues/5283)'
    destination_queue_name = f'destination-queue-{short_uid()}'
    destination_url = sqs_create_queue(QueueName=destination_queue_name)
    snapshot.match('get_destination_queue_url', aws_client.sqs.get_queue_url(QueueName=destination_queue_name))
    retry_timeout = 8
    retries = 2
    function_name = f'failing-lambda-{short_uid()}'
    create_lambda_function(func_name=function_name, handler_file=LAMBDA_SQS_BATCH_ITEM_FAILURE_FILE, runtime=Runtime.python3_8, role=lambda_su_role, timeout=retry_timeout, envvars={'DESTINATION_QUEUE_URL': destination_url})
    event_dlq_url = sqs_create_queue(QueueName=f'event-dlq-{short_uid()}.fifo', Attributes={'FifoQueue': 'true'})
    event_dlq_arn = sqs_get_queue_arn(event_dlq_url)
    event_source_url = sqs_create_queue(QueueName=f'source-queue-{short_uid()}.fifo', Attributes={'FifoQueue': 'true', 'VisibilityTimeout': str(retry_timeout), 'RedrivePolicy': json.dumps({'deadLetterTargetArn': event_dlq_arn, 'maxReceiveCount': retries})})
    event_source_arn = sqs_get_queue_arn(event_source_url)
    response = aws_client.sqs.send_message_batch(QueueUrl=event_source_url, Entries=[{'Id': 'message-1', 'MessageBody': json.dumps({'message': 1, 'fail_attempts': 0}), 'MessageGroupId': '1', 'MessageDeduplicationId': 'dedup-1'}, {'Id': 'message-2', 'MessageBody': json.dumps({'message': 2, 'fail_attempts': 1}), 'MessageGroupId': '1', 'MessageDeduplicationId': 'dedup-2'}, {'Id': 'message-3', 'MessageBody': json.dumps({'message': 3, 'fail_attempts': 1}), 'MessageGroupId': '1', 'MessageDeduplicationId': 'dedup-3'}, {'Id': 'message-4', 'MessageBody': json.dumps({'message': 4, 'fail_attempts': retries}), 'MessageGroupId': '1', 'MessageDeduplicationId': 'dedup-4'}])
    response['Successful'].sort(key=lambda r: r['Id'])
    snapshot.match('send_message_batch', response)
    _await_queue_size(aws_client.sqs, event_source_url, qsize=4, retries=30)
    response = aws_client.lambda_.create_event_source_mapping(EventSourceArn=event_source_arn, FunctionName=function_name, BatchSize=10, MaximumBatchingWindowInSeconds=0, FunctionResponseTypes=['ReportBatchItemFailures'])
    snapshot.match('create_event_source_mapping', response)
    mapping_uuid = response['UUID']
    cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
    _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
    first_invocation = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=int(retry_timeout / 2), MaxNumberOfMessages=1)
    assert 'Messages' in first_invocation
    first_invocation['Messages'][0]['Body'] = json.loads(first_invocation['Messages'][0]['Body'])
    first_invocation['Messages'][0]['Body']['event']['Records'].sort(key=lambda record: json.loads(record['body'])['message'])
    snapshot.match('first_invocation', first_invocation)
    dlq_messages = aws_client.sqs.receive_message(QueueUrl=event_dlq_url)['Messages']
    assert dlq_messages == []
    assert not dlq_messages
    second_invocation = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=retry_timeout + 2, MaxNumberOfMessages=1)
    assert 'Messages' in second_invocation
    second_invocation['Messages'][0]['Body'] = json.loads(second_invocation['Messages'][0]['Body'])
    second_invocation['Messages'][0]['Body']['event']['Records'].sort(key=lambda record: json.loads(record['body'])['message'])
    snapshot.match('second_invocation', second_invocation)
    third_attempt = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=1, MaxNumberOfMessages=1)
    assert third_attempt['Messages'] == []
    dlq_response = aws_client.sqs.receive_message(QueueUrl=event_dlq_url, WaitTimeSeconds=15)
    assert 'Messages' in dlq_response
    snapshot.match('dlq_response', dlq_response)

@markers.aws.validated
def test_report_batch_item_failures_on_lambda_error(create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
    if False:
        while True:
            i = 10
    retry_timeout = 2
    retries = 2
    function_name = f'failing-lambda-{short_uid()}'
    create_lambda_function(func_name=function_name, handler_file=LAMBDA_SQS_INTEGRATION_FILE, runtime=Runtime.python3_8, role=lambda_su_role, timeout=retry_timeout)
    event_dlq_url = sqs_create_queue(QueueName=f'event-dlq-{short_uid()}')
    event_dlq_arn = sqs_get_queue_arn(event_dlq_url)
    event_source_url = sqs_create_queue(QueueName=f'source-queue-{short_uid()}', Attributes={'VisibilityTimeout': str(retry_timeout), 'RedrivePolicy': json.dumps({'deadLetterTargetArn': event_dlq_arn, 'maxReceiveCount': retries})})
    event_source_arn = sqs_get_queue_arn(event_source_url)
    aws_client.sqs.send_message_batch(QueueUrl=event_source_url, Entries=[{'Id': 'message-1', 'MessageBody': '{not a json body'}, {'Id': 'message-2', 'MessageBody': json.dumps({'message': 2, 'fail_attempts': 0})}])
    _await_queue_size(aws_client.sqs, event_source_url, qsize=2)
    mapping_uuid = aws_client.lambda_.create_event_source_mapping(EventSourceArn=event_source_arn, FunctionName=function_name, FunctionResponseTypes=['ReportBatchItemFailures'])['UUID']
    cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
    _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
    messages = []

    def _collect_message():
        if False:
            while True:
                i = 10
        dlq_response = aws_client.sqs.receive_message(QueueUrl=event_dlq_url)
        messages.extend(dlq_response.get('Messages', []))
        assert len(messages) >= 2
    wait_time = retry_timeout * retries
    retry(_collect_message, retries=10, sleep=1, sleep_before=wait_time)
    messages.sort(key=lambda m: m['MD5OfBody'])
    snapshot.match('dlq_messages', messages)

@markers.aws.validated
def test_report_batch_item_failures_invalid_result_json_batch_fails(create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
    if False:
        for i in range(10):
            print('nop')
    destination_queue_name = f'destination-queue-{short_uid()}'
    destination_url = sqs_create_queue(QueueName=destination_queue_name)
    snapshot.match('get_destination_queue_url', aws_client.sqs.get_queue_url(QueueName=destination_queue_name))
    retry_timeout = 4
    retries = 2
    function_name = f'failing-lambda-{short_uid()}'
    create_lambda_function(func_name=function_name, handler_file=LAMBDA_SQS_BATCH_ITEM_FAILURE_FILE, runtime=Runtime.python3_8, role=lambda_su_role, timeout=retry_timeout, envvars={'DESTINATION_QUEUE_URL': destination_url, 'OVERWRITE_RESULT': '{"batchItemFailures": [{"foo":"notvalid"}]}'})
    event_dlq_url = sqs_create_queue(QueueName=f'event-dlq-{short_uid()}')
    event_dlq_arn = sqs_get_queue_arn(event_dlq_url)
    event_source_url = sqs_create_queue(QueueName=f'source-queue-{short_uid()}', Attributes={'VisibilityTimeout': str(retry_timeout), 'RedrivePolicy': json.dumps({'deadLetterTargetArn': event_dlq_arn, 'maxReceiveCount': retries})})
    event_source_arn = sqs_get_queue_arn(event_source_url)
    mapping_uuid = aws_client.lambda_.create_event_source_mapping(EventSourceArn=event_source_arn, FunctionName=function_name, BatchSize=10, MaximumBatchingWindowInSeconds=0, FunctionResponseTypes=['ReportBatchItemFailures'])['UUID']
    cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
    _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
    aws_client.sqs.send_message(QueueUrl=event_source_url, MessageBody=json.dumps({'message': 1, 'fail_attempts': 0}))
    first_invocation = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=15, MaxNumberOfMessages=1)
    assert 'Messages' in first_invocation
    snapshot.match('first_invocation', first_invocation)
    second_invocation = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=15, MaxNumberOfMessages=1)
    assert 'Messages' in second_invocation
    snapshot.match('second_invocation', second_invocation)
    dlq_response = aws_client.sqs.receive_message(QueueUrl=event_dlq_url, WaitTimeSeconds=15)
    assert 'Messages' in dlq_response
    snapshot.match('dlq_response', dlq_response)

@markers.aws.validated
def test_report_batch_item_failures_empty_json_batch_succeeds(create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
    if False:
        print('Hello World!')
    destination_queue_name = f'destination-queue-{short_uid()}'
    destination_url = sqs_create_queue(QueueName=destination_queue_name)
    snapshot.match('get_destination_queue_url', aws_client.sqs.get_queue_url(QueueName=destination_queue_name))
    retry_timeout = 4
    retries = 1
    function_name = f'failing-lambda-{short_uid()}'
    create_lambda_function(func_name=function_name, handler_file=LAMBDA_SQS_BATCH_ITEM_FAILURE_FILE, runtime=Runtime.python3_8, role=lambda_su_role, timeout=retry_timeout, envvars={'DESTINATION_QUEUE_URL': destination_url, 'OVERWRITE_RESULT': '{}'})
    event_dlq_url = sqs_create_queue(QueueName=f'event-dlq-{short_uid()}')
    event_dlq_arn = sqs_get_queue_arn(event_dlq_url)
    event_source_url = sqs_create_queue(QueueName=f'source-queue-{short_uid()}', Attributes={'VisibilityTimeout': str(retry_timeout), 'RedrivePolicy': json.dumps({'deadLetterTargetArn': event_dlq_arn, 'maxReceiveCount': retries})})
    event_source_arn = sqs_get_queue_arn(event_source_url)
    mapping_uuid = aws_client.lambda_.create_event_source_mapping(EventSourceArn=event_source_arn, FunctionName=function_name, BatchSize=10, MaximumBatchingWindowInSeconds=0, FunctionResponseTypes=['ReportBatchItemFailures'])['UUID']
    cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
    _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
    aws_client.sqs.send_message(QueueUrl=event_source_url, MessageBody=json.dumps({'message': 1, 'fail_attempts': 0}))
    first_invocation = aws_client.sqs.receive_message(QueueUrl=destination_url, WaitTimeSeconds=15, MaxNumberOfMessages=1)
    assert 'Messages' in first_invocation
    snapshot.match('first_invocation', first_invocation)
    dlq_response = aws_client.sqs.receive_message(QueueUrl=event_dlq_url, WaitTimeSeconds=retry_timeout + 1)
    assert 'Messages' in dlq_response
    assert dlq_response['Messages'] == []

@markers.snapshot.skip_snapshot_verify(paths=['$..FunctionResponseTypes', '$..LastProcessingResult', '$..MaximumBatchingWindowInSeconds', '$..MaximumRetryAttempts', '$..ParallelizationFactor', '$..ResponseMetadata.HTTPStatusCode', '$..StartingPosition', '$..State', '$..StateTransitionReason', '$..Topics', '$..Records..md5OfMessageAttributes'])
class TestSQSEventSourceMapping:

    @markers.aws.validated
    def test_event_source_mapping_default_batch_size(self, create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, aws_client):
        if False:
            for i in range(10):
                print('nop')
        snapshot.add_transformer(snapshot.transform.lambda_api())
        function_name = f'lambda_func-{short_uid()}'
        queue_name_1 = f'queue-{short_uid()}-1'
        queue_name_2 = f'queue-{short_uid()}-2'
        queue_url_1 = sqs_create_queue(QueueName=queue_name_1)
        queue_arn_1 = sqs_get_queue_arn(queue_url_1)
        try:
            create_lambda_function(func_name=function_name, handler_file=TEST_LAMBDA_PYTHON_ECHO, runtime=Runtime.python3_9, role=lambda_su_role)
            rs = aws_client.lambda_.create_event_source_mapping(EventSourceArn=queue_arn_1, FunctionName=function_name)
            snapshot.match('create-event-source-mapping', rs)
            uuid = rs['UUID']
            assert DEFAULT_SQS_BATCH_SIZE == rs['BatchSize']
            _await_event_source_mapping_enabled(aws_client.lambda_, uuid)
            with pytest.raises(ClientError) as e:
                rs = aws_client.lambda_.update_event_source_mapping(UUID=uuid, FunctionName=function_name, BatchSize=MAX_SQS_BATCH_SIZE_FIFO + 1)
            snapshot.match('invalid-update-event-source-mapping', e.value.response)
            e.match(InvalidParameterValueException.code)
            queue_url_2 = sqs_create_queue(QueueName=queue_name_2)
            queue_arn_2 = sqs_get_queue_arn(queue_url_2)
            with pytest.raises(ClientError) as e:
                rs = aws_client.lambda_.create_event_source_mapping(EventSourceArn=queue_arn_2, FunctionName=function_name, BatchSize=MAX_SQS_BATCH_SIZE_FIFO + 1)
            snapshot.match('invalid-create-event-source-mapping', e.value.response)
            e.match(InvalidParameterValueException.code)
        finally:
            aws_client.lambda_.delete_event_source_mapping(UUID=uuid)

    @markers.aws.validated
    def test_sqs_event_source_mapping(self, create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
        if False:
            i = 10
            return i + 15
        function_name = f'lambda_func-{short_uid()}'
        queue_name_1 = f'queue-{short_uid()}-1'
        mapping_uuid = None
        create_lambda_function(func_name=function_name, handler_file=TEST_LAMBDA_PYTHON_ECHO, runtime=Runtime.python3_9, role=lambda_su_role)
        queue_url_1 = sqs_create_queue(QueueName=queue_name_1)
        queue_arn_1 = sqs_get_queue_arn(queue_url_1)
        create_event_source_mapping_response = aws_client.lambda_.create_event_source_mapping(EventSourceArn=queue_arn_1, FunctionName=function_name, MaximumBatchingWindowInSeconds=1)
        mapping_uuid = create_event_source_mapping_response['UUID']
        cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
        snapshot.match('create-event-source-mapping-response', create_event_source_mapping_response)
        _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
        aws_client.sqs.send_message(QueueUrl=queue_url_1, MessageBody=json.dumps({'foo': 'bar'}))
        events = retry(check_expected_lambda_log_events_length, retries=10, sleep=1, function_name=function_name, expected_length=1, logs_client=aws_client.logs)
        snapshot.match('events', events)
        rs = aws_client.sqs.receive_message(QueueUrl=queue_url_1)
        assert rs.get('Messages') == []

    @markers.aws.validated
    @pytest.mark.parametrize('filter, item_matching, item_not_matching', [({'body': {'testItem': ['test24']}}, {'testItem': 'test24'}, {'testItem': 'tesWER'}), ({'body': {'testItem': ['test24', 'test45']}}, {'testItem': 'test45'}, {'testItem': 'WERTD'}), ({'body': {'testItem': ['test24', 'test45'], 'test2': ['go']}}, {'testItem': 'test45', 'test2': 'go'}, {'testItem': 'test67', 'test2': 'go'}), ({'body': {'test2': [{'exists': True}]}}, {'test2': '7411'}, {'test5': '74545'}), ({'body': {'test2': [{'numeric': ['>', 100]}]}}, {'test2': 105}, 'this is a test string'), ({'body': {'test2': [{'numeric': ['<', 100]}]}}, {'test2': 93}, {'test2': 105}), ({'body': {'test2': [{'numeric': ['>=', 100, '<', 200]}]}}, {'test2': 105}, {'test2': 200}), ({'body': {'test2': [{'prefix': 'us-1'}]}}, {'test2': 'us-1-48454'}, {'test2': 'eu-wert'})])
    def test_sqs_event_filter(self, create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, filter, item_matching, item_not_matching, snapshot, cleanups, aws_client):
        if False:
            for i in range(10):
                print('nop')
        function_name = f'lambda_func-{short_uid()}'
        queue_name_1 = f'queue-{short_uid()}-1'
        mapping_uuid = None
        create_lambda_function(func_name=function_name, handler_file=TEST_LAMBDA_PYTHON_ECHO, runtime=Runtime.python3_9, role=lambda_su_role)
        queue_url_1 = sqs_create_queue(QueueName=queue_name_1)
        queue_arn_1 = sqs_get_queue_arn(queue_url_1)
        aws_client.sqs.send_message(QueueUrl=queue_url_1, MessageBody=json.dumps(item_matching))
        aws_client.sqs.send_message(QueueUrl=queue_url_1, MessageBody=json.dumps(item_not_matching) if not isinstance(item_not_matching, str) else item_not_matching)

        def _assert_qsize():
            if False:
                for i in range(10):
                    print('nop')
            response = aws_client.sqs.get_queue_attributes(QueueUrl=queue_url_1, AttributeNames=['ApproximateNumberOfMessages'])
            assert int(response['Attributes']['ApproximateNumberOfMessages']) == 2
        retry(_assert_qsize, retries=10)
        create_event_source_mapping_response = aws_client.lambda_.create_event_source_mapping(EventSourceArn=queue_arn_1, FunctionName=function_name, MaximumBatchingWindowInSeconds=1, FilterCriteria={'Filters': [{'Pattern': json.dumps(filter)}]})
        mapping_uuid = create_event_source_mapping_response['UUID']
        cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
        snapshot.match('create_event_source_mapping_response', create_event_source_mapping_response)
        _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)

        def _check_lambda_logs():
            if False:
                i = 10
                return i + 15
            events = get_lambda_log_events(function_name, logs_client=aws_client.logs)
            assert len(events) == 1
            records = events[0]['Records']
            assert len(records) == 1
            if 'body' in json.dumps(filter):
                item_matching_str = json.dumps(item_matching)
                assert records[0]['body'] == item_matching_str
            return events
        invocation_events = retry(_check_lambda_logs, retries=10)
        snapshot.match('invocation_events', invocation_events)
        rs = aws_client.sqs.receive_message(QueueUrl=queue_url_1)
        assert rs.get('Messages') == []

    @markers.aws.validated
    @pytest.mark.parametrize('invalid_filter', [None, 'simple string', {'eventSource': 'aws:sqs'}, {'eventSource': []}])
    def test_sqs_invalid_event_filter(self, create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, invalid_filter, snapshot, aws_client):
        if False:
            return 10
        function_name = f'lambda_func-{short_uid()}'
        queue_name_1 = f'queue-{short_uid()}'
        create_lambda_function(func_name=function_name, handler_file=TEST_LAMBDA_PYTHON_ECHO, runtime=Runtime.python3_9, role=lambda_su_role)
        queue_url_1 = sqs_create_queue(QueueName=queue_name_1)
        queue_arn_1 = sqs_get_queue_arn(queue_url_1)
        with pytest.raises(ClientError) as expected:
            aws_client.lambda_.create_event_source_mapping(EventSourceArn=queue_arn_1, FunctionName=function_name, MaximumBatchingWindowInSeconds=1, FilterCriteria={'Filters': [{'Pattern': invalid_filter if isinstance(invalid_filter, str) else json.dumps(invalid_filter)}]})
        snapshot.match('create_event_source_mapping_exception', expected.value.response)
        expected.match(InvalidParameterValueException.code)

    @markers.aws.validated
    def test_sqs_event_source_mapping_update(self, create_lambda_function, sqs_create_queue, sqs_get_queue_arn, lambda_su_role, snapshot, cleanups, aws_client):
        if False:
            for i in range(10):
                print('nop')
        '\n        Testing an update to an event source mapping that changes the targeted lambda function version\n\n        Resources used:\n        - Lambda function\n        - 2 published versions of that lambda function\n        - 1 event source mapping\n\n        First the event source mapping points towards the qualified ARN of the first version.\n        A message is sent to the SQS queue, triggering the function version with ID 1.\n        The lambda function is updated with a different value for the environment variable and a new version published.\n        Then we update the event source mapping and make the qualified ARN of the function version with ID 2 the new target.\n        A message is sent to the SQS queue, triggering the function with version ID 2.\n\n        We should have one log entry for each of the invocations.\n\n        '
        function_name = f'lambda_func-{short_uid()}'
        queue_name_1 = f'queue-{short_uid()}-1'
        mapping_uuid = None
        create_lambda_function(func_name=function_name, handler_file=TEST_LAMBDA_PYTHON_ECHO_VERSION_ENV, runtime=Runtime.python3_11, role=lambda_su_role)
        aws_client.lambda_.update_function_configuration(FunctionName=function_name, Environment={'Variables': {'CUSTOM_VAR': 'a'}})
        aws_client.lambda_.get_waiter('function_updated_v2').wait(FunctionName=function_name)
        publish_v1 = aws_client.lambda_.publish_version(FunctionName=function_name)
        aws_client.lambda_.get_waiter('function_active_v2').wait(FunctionName=publish_v1['FunctionArn'])
        queue_url_1 = sqs_create_queue(QueueName=queue_name_1)
        queue_arn_1 = sqs_get_queue_arn(queue_url_1)
        create_event_source_mapping_response = aws_client.lambda_.create_event_source_mapping(EventSourceArn=queue_arn_1, FunctionName=publish_v1['FunctionArn'], MaximumBatchingWindowInSeconds=1)
        mapping_uuid = create_event_source_mapping_response['UUID']
        cleanups.append(lambda : aws_client.lambda_.delete_event_source_mapping(UUID=mapping_uuid))
        snapshot.match('create-event-source-mapping-response', create_event_source_mapping_response)
        _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
        aws_client.sqs.send_message(QueueUrl=queue_url_1, MessageBody=json.dumps({'foo': 'bar'}))
        events = retry(check_expected_lambda_log_events_length, retries=10, sleep=1, function_name=function_name, expected_length=1, logs_client=aws_client.logs)
        snapshot.match('events', events)
        rs = aws_client.sqs.receive_message(QueueUrl=queue_url_1)
        assert rs.get('Messages') == []
        aws_client.lambda_.update_function_configuration(FunctionName=function_name, Environment={'Variables': {'CUSTOM_VAR': 'b'}})
        aws_client.lambda_.get_waiter('function_updated_v2').wait(FunctionName=function_name)
        publish_v2 = aws_client.lambda_.publish_version(FunctionName=function_name)
        aws_client.lambda_.get_waiter('function_active_v2').wait(FunctionName=publish_v2['FunctionArn'])
        updated_esm = aws_client.lambda_.update_event_source_mapping(UUID=mapping_uuid, FunctionName=publish_v2['FunctionArn'])
        assert mapping_uuid == updated_esm['UUID']
        assert publish_v2['FunctionArn'] == updated_esm['FunctionArn']
        snapshot.match('updated_esm', updated_esm)
        _await_event_source_mapping_enabled(aws_client.lambda_, mapping_uuid)
        if is_aws_cloud():
            time.sleep(10)
        aws_client.sqs.send_message(QueueUrl=queue_url_1, MessageBody=json.dumps({'foo': 'bar2'}))
        events_postupdate = retry(check_expected_lambda_log_events_length, retries=10, sleep=1, function_name=function_name, expected_length=2, logs_client=aws_client.logs)
        snapshot.match('events_postupdate', events_postupdate)
        rs = aws_client.sqs.receive_message(QueueUrl=queue_url_1)
        assert rs.get('Messages') == []