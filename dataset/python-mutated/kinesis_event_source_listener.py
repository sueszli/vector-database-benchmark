import base64
from typing import Dict, List, Optional
from localstack.services.lambda_.event_source_listeners.stream_event_source_listener import StreamEventSourceListener
from localstack.utils.aws.arns import extract_account_id_from_arn, extract_region_from_arn
from localstack.utils.common import first_char_to_lower, to_str
from localstack.utils.threads import FuncThread

class KinesisEventSourceListener(StreamEventSourceListener):
    _FAILURE_PAYLOAD_DETAILS_FIELD_NAME = 'KinesisBatchInfo'
    _COORDINATOR_THREAD: Optional[FuncThread] = None
    _STREAM_LISTENER_THREADS: Dict[str, FuncThread] = {}

    @staticmethod
    def source_type() -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return 'kinesis'

    def _get_matching_event_sources(self) -> List[Dict]:
        if False:
            i = 10
            return i + 15
        event_sources = self._invoke_adapter.get_event_sources(source_arn='.*:kinesis:.*')
        return [source for source in event_sources if source['State'] == 'Enabled']

    def _get_stream_client(self, function_arn: str, region_name: str):
        if False:
            print('Hello World!')
        return self._invoke_adapter.get_client_factory(function_arn=function_arn, region_name=region_name).kinesis.request_metadata(source_arn=function_arn)

    def _get_stream_description(self, stream_client, stream_arn):
        if False:
            print('Hello World!')
        stream_name = stream_arn.split('/')[-1]
        return stream_client.describe_stream(StreamName=stream_name)['StreamDescription']

    def _get_shard_iterator(self, stream_client, stream_arn, shard_id, iterator_type):
        if False:
            for i in range(10):
                print('nop')
        stream_name = stream_arn.split('/')[-1]
        return stream_client.get_shard_iterator(StreamName=stream_name, ShardId=shard_id, ShardIteratorType=iterator_type)['ShardIterator']

    def _create_lambda_event_payload(self, stream_arn, records, shard_id=None):
        if False:
            return 10
        record_payloads = []
        for record in records:
            record_payload = {}
            for (key, val) in record.items():
                record_payload[first_char_to_lower(key)] = val
            record_payload['data'] = to_str(base64.b64encode(record_payload['data']))
            record_payload['approximateArrivalTimestamp'] = record_payload['approximateArrivalTimestamp'].timestamp()
            record_payload.pop('encryptionType', None)
            record_payloads.append({'eventID': '{0}:{1}'.format(shard_id, record_payload['sequenceNumber']), 'eventSourceARN': stream_arn, 'eventSource': 'aws:kinesis', 'eventVersion': '1.0', 'eventName': 'aws:kinesis:record', 'invokeIdentityArn': f'arn:aws:iam::{extract_account_id_from_arn(stream_arn)}:role/lambda-role', 'awsRegion': extract_region_from_arn(stream_arn), 'kinesis': record_payload})
        return {'Records': record_payloads}

    def _get_starting_and_ending_sequence_numbers(self, first_record, last_record):
        if False:
            while True:
                i = 10
        return (first_record['SequenceNumber'], last_record['SequenceNumber'])

    def _get_first_and_last_arrival_time(self, first_record, last_record):
        if False:
            print('Hello World!')
        return (first_record['ApproximateArrivalTimestamp'].isoformat() + 'Z', last_record['ApproximateArrivalTimestamp'].isoformat() + 'Z')