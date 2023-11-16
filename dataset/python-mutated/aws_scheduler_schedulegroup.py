from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class SchedulerScheduleGroupProperties(TypedDict):
    Arn: Optional[str]
    CreationDate: Optional[str]
    LastModificationDate: Optional[str]
    Name: Optional[str]
    State: Optional[str]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class SchedulerScheduleGroupProvider(ResourceProvider[SchedulerScheduleGroupProperties]):
    TYPE = 'AWS::Scheduler::ScheduleGroup'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SchedulerScheduleGroupProperties]) -> ProgressEvent[SchedulerScheduleGroupProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Name\n\n\n\n        Create-only properties:\n          - /properties/Name\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/CreationDate\n          - /properties/LastModificationDate\n          - /properties/State\n\n        IAM permissions required:\n          - scheduler:CreateScheduleGroup\n          - scheduler:GetScheduleGroup\n          - scheduler:ListTagsForResource\n\n        '
        model = request.desired_state
        if not model.get('Name'):
            model['Name'] = util.generate_default_name(request.stack_name, request.logical_resource_id)
        create_params = util.select_attributes(model, ('Name', 'Tags'))
        result = request.aws_client_factory.scheduler.create_schedule_group(**create_params)
        model['Arn'] = result['ScheduleGroupArn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[SchedulerScheduleGroupProperties]) -> ProgressEvent[SchedulerScheduleGroupProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - scheduler:GetScheduleGroup\n          - scheduler:ListTagsForResource\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SchedulerScheduleGroupProperties]) -> ProgressEvent[SchedulerScheduleGroupProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - scheduler:DeleteScheduleGroup\n          - scheduler:GetScheduleGroup\n          - scheduler:DeleteSchedule\n        '
        model = request.desired_state
        request.aws_client_factory.scheduler.delete_schedule_group(Name=model['Name'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[SchedulerScheduleGroupProperties]) -> ProgressEvent[SchedulerScheduleGroupProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n        IAM permissions required:\n          - scheduler:TagResource\n          - scheduler:UntagResource\n          - scheduler:ListTagsForResource\n          - scheduler:GetScheduleGroup\n        '
        raise NotImplementedError