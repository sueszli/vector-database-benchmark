import datetime
from typing import Dict, List, Optional
from localstack.services.lambda_.event_source_listeners.stream_event_source_listener import StreamEventSourceListener
from localstack.utils.aws import aws_stack
from localstack.utils.threads import FuncThread

class DynamoDBEventSourceListener(StreamEventSourceListener):
    _FAILURE_PAYLOAD_DETAILS_FIELD_NAME = 'DDBStreamBatchInfo'
    _COORDINATOR_THREAD: Optional[FuncThread] = None
    _STREAM_LISTENER_THREADS: Dict[str, FuncThread] = {}

    @staticmethod
    def source_type() -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return 'dynamodb'

    def _get_matching_event_sources(self) -> List[Dict]:
        if False:
            print('Hello World!')
        event_sources = self._invoke_adapter.get_event_sources(source_arn='.*:dynamodb:.*')
        return [source for source in event_sources if source['State'] == 'Enabled']

    def _get_stream_client(self, function_arn: str, region_name: str):
        if False:
            return 10
        return self._invoke_adapter.get_client_factory(function_arn=function_arn, region_name=region_name).dynamodbstreams.request_metadata(source_arn=function_arn)

    def _get_stream_description(self, stream_client, stream_arn):
        if False:
            for i in range(10):
                print('nop')
        return stream_client.describe_stream(StreamArn=stream_arn)['StreamDescription']

    def _get_shard_iterator(self, stream_client, stream_arn, shard_id, iterator_type):
        if False:
            print('Hello World!')
        return stream_client.get_shard_iterator(StreamArn=stream_arn, ShardId=shard_id, ShardIteratorType=iterator_type)['ShardIterator']

    def _create_lambda_event_payload(self, stream_arn, records, shard_id=None):
        if False:
            return 10
        record_payloads = []
        for record in records:
            creation_time = record.get('dynamodb', {}).get('ApproximateCreationDateTime', None)
            if creation_time is not None:
                record['dynamodb']['ApproximateCreationDateTime'] = creation_time.timestamp()
            record_payloads.append({'eventID': record['eventID'], 'eventVersion': '1.0', 'awsRegion': aws_stack.get_region(), 'eventName': record['eventName'], 'eventSourceARN': stream_arn, 'eventSource': 'aws:dynamodb', 'dynamodb': record['dynamodb']})
        return {'Records': record_payloads}

    def _get_starting_and_ending_sequence_numbers(self, first_record, last_record):
        if False:
            return 10
        return (first_record['dynamodb']['SequenceNumber'], last_record['dynamodb']['SequenceNumber'])

    def _get_first_and_last_arrival_time(self, first_record, last_record):
        if False:
            print('Hello World!')
        return (first_record.get('ApproximateArrivalTimestamp', datetime.datetime.utcnow()).isoformat() + 'Z', last_record.get('ApproximateArrivalTimestamp', datetime.datetime.utcnow()).isoformat() + 'Z')