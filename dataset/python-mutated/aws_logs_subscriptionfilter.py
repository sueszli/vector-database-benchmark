from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class LogsSubscriptionFilterProperties(TypedDict):
    DestinationArn: Optional[str]
    FilterPattern: Optional[str]
    LogGroupName: Optional[str]
    Distribution: Optional[str]
    FilterName: Optional[str]
    RoleArn: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class LogsSubscriptionFilterProvider(ResourceProvider[LogsSubscriptionFilterProperties]):
    TYPE = 'AWS::Logs::SubscriptionFilter'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[LogsSubscriptionFilterProperties]) -> ProgressEvent[LogsSubscriptionFilterProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/FilterName\n          - /properties/LogGroupName\n\n        Required properties:\n          - DestinationArn\n          - FilterPattern\n          - LogGroupName\n\n        Create-only properties:\n          - /properties/FilterName\n          - /properties/LogGroupName\n\n\n\n        IAM permissions required:\n          - iam:PassRole\n          - logs:PutSubscriptionFilter\n          - logs:DescribeSubscriptionFilters\n\n        '
        model = request.desired_state
        logs = request.aws_client_factory.logs
        logs.put_subscription_filter(logGroupName=model['LogGroupName'], filterName=model['LogGroupName'], filterPattern=model['FilterPattern'], destinationArn=model['DestinationArn'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[LogsSubscriptionFilterProperties]) -> ProgressEvent[LogsSubscriptionFilterProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - logs:DescribeSubscriptionFilters\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[LogsSubscriptionFilterProperties]) -> ProgressEvent[LogsSubscriptionFilterProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - logs:DeleteSubscriptionFilter\n        '
        model = request.desired_state
        logs = request.aws_client_factory.logs
        logs.delete_subscription_filter(logGroupName=model['LogGroupName'], filterName=model['LogGroupName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[LogsSubscriptionFilterProperties]) -> ProgressEvent[LogsSubscriptionFilterProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n\n        IAM permissions required:\n          - logs:PutSubscriptionFilter\n          - logs:DescribeSubscriptionFilters\n        '
        raise NotImplementedError