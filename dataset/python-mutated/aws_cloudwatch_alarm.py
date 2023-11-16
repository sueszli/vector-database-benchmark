from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class CloudWatchAlarmProperties(TypedDict):
    ComparisonOperator: Optional[str]
    EvaluationPeriods: Optional[int]
    ActionsEnabled: Optional[bool]
    AlarmActions: Optional[list[str]]
    AlarmDescription: Optional[str]
    AlarmName: Optional[str]
    Arn: Optional[str]
    DatapointsToAlarm: Optional[int]
    Dimensions: Optional[list[Dimension]]
    EvaluateLowSampleCountPercentile: Optional[str]
    ExtendedStatistic: Optional[str]
    Id: Optional[str]
    InsufficientDataActions: Optional[list[str]]
    MetricName: Optional[str]
    Metrics: Optional[list[MetricDataQuery]]
    Namespace: Optional[str]
    OKActions: Optional[list[str]]
    Period: Optional[int]
    Statistic: Optional[str]
    Threshold: Optional[float]
    ThresholdMetricId: Optional[str]
    TreatMissingData: Optional[str]
    Unit: Optional[str]

class Dimension(TypedDict):
    Name: Optional[str]
    Value: Optional[str]

class Metric(TypedDict):
    Dimensions: Optional[list[Dimension]]
    MetricName: Optional[str]
    Namespace: Optional[str]

class MetricStat(TypedDict):
    Metric: Optional[Metric]
    Period: Optional[int]
    Stat: Optional[str]
    Unit: Optional[str]

class MetricDataQuery(TypedDict):
    Id: Optional[str]
    AccountId: Optional[str]
    Expression: Optional[str]
    Label: Optional[str]
    MetricStat: Optional[MetricStat]
    Period: Optional[int]
    ReturnData: Optional[bool]
REPEATED_INVOCATION = 'repeated_invocation'

class CloudWatchAlarmProvider(ResourceProvider[CloudWatchAlarmProperties]):
    TYPE = 'AWS::CloudWatch::Alarm'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[CloudWatchAlarmProperties]) -> ProgressEvent[CloudWatchAlarmProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - ComparisonOperator\n          - EvaluationPeriods\n\n        Create-only properties:\n          - /properties/AlarmName\n\n        Read-only properties:\n          - /properties/Id\n          - /properties/Arn\n\n\n\n        '
        model = request.desired_state
        cloud_watch = request.aws_client_factory.cloudwatch
        if not model.get('AlarmName'):
            model['AlarmName'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        cloud_watch.put_metric_alarm(AlarmName=model['AlarmName'], ComparisonOperator=model['ComparisonOperator'], EvaluationPeriods=model['EvaluationPeriods'], Period=model['Period'], MetricName=model['MetricName'], Namespace=model['Namespace'], Statistic=model['Statistic'], Threshold=model['Threshold'])
        alarm = cloud_watch.describe_alarms(AlarmNames=[model['AlarmName']])['MetricAlarms'][0]
        model['Arn'] = alarm['AlarmArn']
        model['Id'] = alarm['AlarmName']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[CloudWatchAlarmProperties]) -> ProgressEvent[CloudWatchAlarmProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[CloudWatchAlarmProperties]) -> ProgressEvent[CloudWatchAlarmProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        cloud_watch = request.aws_client_factory.cloudwatch
        cloud_watch.delete_alarms(AlarmNames=[model['AlarmName']])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[CloudWatchAlarmProperties]) -> ProgressEvent[CloudWatchAlarmProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError