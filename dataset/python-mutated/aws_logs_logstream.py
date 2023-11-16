from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class LogsLogStreamProperties(TypedDict):
    LogGroupName: Optional[str]
    Id: Optional[str]
    LogStreamName: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class LogsLogStreamProvider(ResourceProvider[LogsLogStreamProperties]):
    TYPE = 'AWS::Logs::LogStream'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[LogsLogStreamProperties]) -> ProgressEvent[LogsLogStreamProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - LogGroupName\n\n        Create-only properties:\n          - /properties/LogGroupName\n          - /properties/LogStreamName\n\n        Read-only properties:\n          - /properties/Id\n\n        '
        model = request.desired_state
        logs = request.aws_client_factory.logs
        if not model.get('LogStreamName'):
            model['LogStreamName'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        logs.create_log_stream(logGroupName=model['LogGroupName'], logStreamName=model['LogStreamName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[LogsLogStreamProperties]) -> ProgressEvent[LogsLogStreamProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[LogsLogStreamProperties]) -> ProgressEvent[LogsLogStreamProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n        '
        model = request.desired_state
        logs = request.aws_client_factory.logs
        logs.delete_log_stream(logGroupName=model['LogGroupName'], logStreamName=model['LogStreamName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[LogsLogStreamProperties]) -> ProgressEvent[LogsLogStreamProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError