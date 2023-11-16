from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class KinesisStreamProperties(TypedDict):
    Arn: Optional[str]
    Name: Optional[str]
    RetentionPeriodHours: Optional[int]
    ShardCount: Optional[int]
    StreamEncryption: Optional[StreamEncryption]
    StreamModeDetails: Optional[StreamModeDetails]
    Tags: Optional[list[Tag]]

class StreamModeDetails(TypedDict):
    StreamMode: Optional[str]

class StreamEncryption(TypedDict):
    EncryptionType: Optional[str]
    KeyId: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class KinesisStreamProvider(ResourceProvider[KinesisStreamProperties]):
    TYPE = 'AWS::Kinesis::Stream'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[KinesisStreamProperties]) -> ProgressEvent[KinesisStreamProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Name\n\n\n\n        Create-only properties:\n          - /properties/Name\n\n        Read-only properties:\n          - /properties/Arn\n\n        IAM permissions required:\n          - kinesis:EnableEnhancedMonitoring\n          - kinesis:DescribeStreamSummary\n          - kinesis:CreateStream\n          - kinesis:IncreaseStreamRetentionPeriod\n          - kinesis:StartStreamEncryption\n          - kinesis:AddTagsToStream\n          - kinesis:ListTagsForStream\n\n        '
        model = request.desired_state
        kinesis = request.aws_client_factory.kinesis
        if not request.custom_context.get(REPEATED_INVOCATION):
            if not model.get('Name'):
                model['Name'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
            if not model.get('ShardCount'):
                model['ShardCount'] = 1
            if not model.get('StreamModeDetails'):
                model['StreamModeDetails'] = StreamModeDetails(StreamMode='ON_DEMAND')
            kinesis.create_stream(StreamName=model['Name'], ShardCount=model['ShardCount'], StreamModeDetails=model['StreamModeDetails'])
            stream_data = kinesis.describe_stream(StreamName=model['Name'])['StreamDescription']
            model['Arn'] = stream_data['StreamARN']
            request.custom_context[REPEATED_INVOCATION] = True
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        stream_data = kinesis.describe_stream(StreamARN=model['Arn'])['StreamDescription']
        if stream_data['StreamStatus'] != 'ACTIVE':
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[KinesisStreamProperties]) -> ProgressEvent[KinesisStreamProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - kinesis:DescribeStreamSummary\n          - kinesis:ListTagsForStream\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[KinesisStreamProperties]) -> ProgressEvent[KinesisStreamProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - kinesis:DescribeStreamSummary\n          - kinesis:DeleteStream\n          - kinesis:RemoveTagsFromStream\n        '
        model = request.desired_state
        request.aws_client_factory.kinesis.delete_stream(StreamARN=model['Arn'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[KinesisStreamProperties]) -> ProgressEvent[KinesisStreamProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n        IAM permissions required:\n          - kinesis:EnableEnhancedMonitoring\n          - kinesis:DisableEnhancedMonitoring\n          - kinesis:DescribeStreamSummary\n          - kinesis:UpdateShardCount\n          - kinesis:UpdateStreamMode\n          - kinesis:IncreaseStreamRetentionPeriod\n          - kinesis:DecreaseStreamRetentionPeriod\n          - kinesis:StartStreamEncryption\n          - kinesis:StopStreamEncryption\n          - kinesis:AddTagsToStream\n          - kinesis:RemoveTagsFromStream\n          - kinesis:ListTagsForStream\n        '
        raise NotImplementedError