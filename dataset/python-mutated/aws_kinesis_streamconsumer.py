from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class KinesisStreamConsumerProperties(TypedDict):
    ConsumerName: Optional[str]
    StreamARN: Optional[str]
    ConsumerARN: Optional[str]
    ConsumerCreationTimestamp: Optional[str]
    ConsumerStatus: Optional[str]
    Id: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class KinesisStreamConsumerProvider(ResourceProvider[KinesisStreamConsumerProperties]):
    TYPE = 'AWS::Kinesis::StreamConsumer'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[KinesisStreamConsumerProperties]) -> ProgressEvent[KinesisStreamConsumerProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - ConsumerName\n          - StreamARN\n\n        Create-only properties:\n          - /properties/ConsumerName\n          - /properties/StreamARN\n\n        Read-only properties:\n          - /properties/ConsumerStatus\n          - /properties/ConsumerARN\n          - /properties/ConsumerCreationTimestamp\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        kinesis = request.aws_client_factory.kinesis
        if not request.custom_context.get(REPEATED_INVOCATION):
            response = kinesis.register_stream_consumer(StreamARN=model['StreamARN'], ConsumerName=model['ConsumerName'])
            model['ConsumerARN'] = response['Consumer']['ConsumerARN']
            model['ConsumerStatus'] = response['Consumer']['ConsumerStatus']
            request.custom_context[REPEATED_INVOCATION] = True
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        response = kinesis.describe_stream_consumer(ConsumerARN=model['ConsumerARN'])
        model['ConsumerStatus'] = response['ConsumerDescription']['ConsumerStatus']
        if model['ConsumerStatus'] == 'CREATING':
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[KinesisStreamConsumerProperties]) -> ProgressEvent[KinesisStreamConsumerProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n\n        '
        model = request.desired_state
        kinesis = request.aws_client_factory.kinesis
        kinesis.deregister_stream_consumer(ConsumerARN=model['ConsumerARN'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def delete(self, request: ResourceRequest[KinesisStreamConsumerProperties]) -> ProgressEvent[KinesisStreamConsumerProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n\n        '
        raise NotImplementedError

    def update(self, request: ResourceRequest[KinesisStreamConsumerProperties]) -> ProgressEvent[KinesisStreamConsumerProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError