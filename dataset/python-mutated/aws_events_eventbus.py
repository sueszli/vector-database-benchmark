from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EventsEventBusProperties(TypedDict):
    Name: Optional[str]
    Arn: Optional[str]
    EventSourceName: Optional[str]
    Id: Optional[str]
    Policy: Optional[str]
    Tags: Optional[list[TagEntry]]

class TagEntry(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EventsEventBusProvider(ResourceProvider[EventsEventBusProperties]):
    TYPE = 'AWS::Events::EventBus'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EventsEventBusProperties]) -> ProgressEvent[EventsEventBusProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - Name\n\n        Create-only properties:\n          - /properties/Name\n          - /properties/EventSourceName\n\n        Read-only properties:\n          - /properties/Id\n          - /properties/Policy\n          - /properties/Arn\n\n        '
        model = request.desired_state
        events = request.aws_client_factory.events
        response = events.create_event_bus(Name=model['Name'])
        model['Arn'] = response['EventBusArn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EventsEventBusProperties]) -> ProgressEvent[EventsEventBusProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EventsEventBusProperties]) -> ProgressEvent[EventsEventBusProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        events = request.aws_client_factory.events
        events.delete_event_bus(Name=model['Name'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EventsEventBusProperties]) -> ProgressEvent[EventsEventBusProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError