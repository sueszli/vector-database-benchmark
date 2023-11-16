from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.strings import str_to_bool

class CloudWatchCompositeAlarmProperties(TypedDict):
    AlarmRule: Optional[str]
    ActionsEnabled: Optional[bool]
    ActionsSuppressor: Optional[str]
    ActionsSuppressorExtensionPeriod: Optional[int]
    ActionsSuppressorWaitPeriod: Optional[int]
    AlarmActions: Optional[list[str]]
    AlarmDescription: Optional[str]
    AlarmName: Optional[str]
    Arn: Optional[str]
    InsufficientDataActions: Optional[list[str]]
    OKActions: Optional[list[str]]
REPEATED_INVOCATION = 'repeated_invocation'

class CloudWatchCompositeAlarmProvider(ResourceProvider[CloudWatchCompositeAlarmProperties]):
    TYPE = 'AWS::CloudWatch::CompositeAlarm'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[CloudWatchCompositeAlarmProperties]) -> ProgressEvent[CloudWatchCompositeAlarmProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/AlarmName\n\n        Required properties:\n          - AlarmRule\n\n        Create-only properties:\n          - /properties/AlarmName\n\n        Read-only properties:\n          - /properties/Arn\n\n        IAM permissions required:\n          - cloudwatch:DescribeAlarms\n          - cloudwatch:PutCompositeAlarm\n\n        '
        model = request.desired_state
        cloud_watch = request.aws_client_factory.cloudwatch
        params = util.select_attributes(model, ['AlarmName', 'AlarmRule', 'ActionsEnabled', 'ActionsSuppressor', 'ActionsSuppressorWaitPeriod', 'ActionsSuppressorExtensionPeriod', 'AlarmActions', 'AlarmDescription', 'InsufficientDataActions', 'OKActions'])
        if not params.get('AlarmName'):
            model['AlarmName'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
            params['AlarmName'] = model['AlarmName']
        if 'ActionsEnabled' in params:
            params['ActionsEnabled'] = str_to_bool(params['ActionsEnabled'])
        cloud_watch.put_composite_alarm(AlarmName=model['AlarmName'], AlarmRule=model['AlarmRule'])
        alarm = cloud_watch.describe_alarms(AlarmNames=[model['AlarmName']])['MetricAlarms'][0]
        model['Arn'] = alarm['AlarmArn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[CloudWatchCompositeAlarmProperties]) -> ProgressEvent[CloudWatchCompositeAlarmProperties]:
        if False:
            return 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - cloudwatch:DescribeAlarms\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[CloudWatchCompositeAlarmProperties]) -> ProgressEvent[CloudWatchCompositeAlarmProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n        IAM permissions required:\n          - cloudwatch:DescribeAlarms\n          - cloudwatch:DeleteAlarms\n        '
        model = request.desired_state
        cloud_watch = request.aws_client_factory.cloudwatch
        cloud_watch.delete_alarms(AlarmNames=[model['AlarmName']])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[CloudWatchCompositeAlarmProperties]) -> ProgressEvent[CloudWatchCompositeAlarmProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n        IAM permissions required:\n          - cloudwatch:DescribeAlarms\n          - cloudwatch:PutCompositeAlarm\n        '
        raise NotImplementedError