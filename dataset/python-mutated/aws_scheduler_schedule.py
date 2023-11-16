from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class SchedulerScheduleProperties(TypedDict):
    FlexibleTimeWindow: Optional[FlexibleTimeWindow]
    ScheduleExpression: Optional[str]
    Target: Optional[Target]
    Arn: Optional[str]
    Description: Optional[str]
    EndDate: Optional[str]
    GroupName: Optional[str]
    KmsKeyArn: Optional[str]
    Name: Optional[str]
    ScheduleExpressionTimezone: Optional[str]
    StartDate: Optional[str]
    State: Optional[str]

class FlexibleTimeWindow(TypedDict):
    Mode: Optional[str]
    MaximumWindowInMinutes: Optional[float]

class DeadLetterConfig(TypedDict):
    Arn: Optional[str]

class RetryPolicy(TypedDict):
    MaximumEventAgeInSeconds: Optional[float]
    MaximumRetryAttempts: Optional[float]

class AwsVpcConfiguration(TypedDict):
    Subnets: Optional[list[str]]
    AssignPublicIp: Optional[str]
    SecurityGroups: Optional[list[str]]

class NetworkConfiguration(TypedDict):
    AwsvpcConfiguration: Optional[AwsVpcConfiguration]

class CapacityProviderStrategyItem(TypedDict):
    CapacityProvider: Optional[str]
    Base: Optional[float]
    Weight: Optional[float]

class PlacementConstraint(TypedDict):
    Expression: Optional[str]
    Type: Optional[str]

class PlacementStrategy(TypedDict):
    Field: Optional[str]
    Type: Optional[str]

class EcsParameters(TypedDict):
    TaskDefinitionArn: Optional[str]
    CapacityProviderStrategy: Optional[list[CapacityProviderStrategyItem]]
    EnableECSManagedTags: Optional[bool]
    EnableExecuteCommand: Optional[bool]
    Group: Optional[str]
    LaunchType: Optional[str]
    NetworkConfiguration: Optional[NetworkConfiguration]
    PlacementConstraints: Optional[list[PlacementConstraint]]
    PlacementStrategy: Optional[list[PlacementStrategy]]
    PlatformVersion: Optional[str]
    PropagateTags: Optional[str]
    ReferenceId: Optional[str]
    Tags: Optional[list[dict]]
    TaskCount: Optional[float]

class EventBridgeParameters(TypedDict):
    DetailType: Optional[str]
    Source: Optional[str]

class KinesisParameters(TypedDict):
    PartitionKey: Optional[str]

class SageMakerPipelineParameter(TypedDict):
    Name: Optional[str]
    Value: Optional[str]

class SageMakerPipelineParameters(TypedDict):
    PipelineParameterList: Optional[list[SageMakerPipelineParameter]]

class SqsParameters(TypedDict):
    MessageGroupId: Optional[str]

class Target(TypedDict):
    Arn: Optional[str]
    RoleArn: Optional[str]
    DeadLetterConfig: Optional[DeadLetterConfig]
    EcsParameters: Optional[EcsParameters]
    EventBridgeParameters: Optional[EventBridgeParameters]
    Input: Optional[str]
    KinesisParameters: Optional[KinesisParameters]
    RetryPolicy: Optional[RetryPolicy]
    SageMakerPipelineParameters: Optional[SageMakerPipelineParameters]
    SqsParameters: Optional[SqsParameters]
REPEATED_INVOCATION = 'repeated_invocation'

class SchedulerScheduleProvider(ResourceProvider[SchedulerScheduleProperties]):
    TYPE = 'AWS::Scheduler::Schedule'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SchedulerScheduleProperties]) -> ProgressEvent[SchedulerScheduleProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Name\n\n        Required properties:\n          - FlexibleTimeWindow\n          - ScheduleExpression\n          - Target\n\n        Create-only properties:\n          - /properties/Name\n\n        Read-only properties:\n          - /properties/Arn\n\n        IAM permissions required:\n          - scheduler:CreateSchedule\n          - scheduler:GetSchedule\n          - iam:PassRole\n\n        '
        model = request.desired_state
        if not model.get('Name'):
            model['Name'] = util.generate_default_name(request.stack_name, request.logical_resource_id)
        create_params = util.select_attributes(model, ['Description', 'EndDate', 'FlexibleTimeWindow', 'GroupName', 'KmsKeyArn', 'Name', 'ScheduleExpression', 'ScheduleExpressionTimezone', 'StartDate', 'State', 'Target'])
        result = request.aws_client_factory.scheduler.create_schedule(**create_params)
        model['Arn'] = result['ScheduleArn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[SchedulerScheduleProperties]) -> ProgressEvent[SchedulerScheduleProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - scheduler:GetSchedule\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SchedulerScheduleProperties]) -> ProgressEvent[SchedulerScheduleProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n        IAM permissions required:\n          - scheduler:DeleteSchedule\n          - scheduler:GetSchedule\n        '
        delete_params = util.select_attributes(request.desired_state, ['Name', 'GroupName'])
        request.aws_client_factory.scheduler.delete_schedule(**delete_params)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[SchedulerScheduleProperties]) -> ProgressEvent[SchedulerScheduleProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - scheduler:UpdateSchedule\n          - scheduler:GetSchedule\n          - iam:PassRole\n        '
        raise NotImplementedError