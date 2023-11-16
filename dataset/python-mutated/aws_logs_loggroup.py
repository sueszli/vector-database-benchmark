from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class LogsLogGroupProperties(TypedDict):
    Arn: Optional[str]
    DataProtectionPolicy: Optional[dict]
    KmsKeyId: Optional[str]
    LogGroupName: Optional[str]
    RetentionInDays: Optional[int]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class LogsLogGroupProvider(ResourceProvider[LogsLogGroupProperties]):
    TYPE = 'AWS::Logs::LogGroup'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[LogsLogGroupProperties]) -> ProgressEvent[LogsLogGroupProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/LogGroupName\n\n        Create-only properties:\n          - /properties/LogGroupName\n\n        Read-only properties:\n          - /properties/Arn\n\n        IAM permissions required:\n          - logs:DescribeLogGroups\n          - logs:CreateLogGroup\n          - logs:PutRetentionPolicy\n          - logs:TagLogGroup\n          - logs:GetDataProtectionPolicy\n          - logs:PutDataProtectionPolicy\n          - logs:CreateLogDelivery\n          - s3:REST.PUT.OBJECT\n          - firehose:TagDeliveryStream\n          - logs:PutResourcePolicy\n          - logs:DescribeResourcePolicies\n\n        '
        model = request.desired_state
        logs = request.aws_client_factory.logs
        if not model.get('LogGroupName'):
            model['LogGroupName'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        logs.create_log_group(logGroupName=model['LogGroupName'])
        describe_result = logs.describe_log_groups(logGroupNamePrefix=model['LogGroupName'])
        model['Arn'] = describe_result['logGroups'][0]['arn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[LogsLogGroupProperties]) -> ProgressEvent[LogsLogGroupProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - logs:DescribeLogGroups\n          - logs:ListTagsLogGroup\n          - logs:GetDataProtectionPolicy\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[LogsLogGroupProperties]) -> ProgressEvent[LogsLogGroupProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - logs:DescribeLogGroups\n          - logs:DeleteLogGroup\n          - logs:DeleteDataProtectionPolicy\n        '
        model = request.desired_state
        logs = request.aws_client_factory.logs
        logs.delete_log_group(logGroupName=model['LogGroupName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[LogsLogGroupProperties]) -> ProgressEvent[LogsLogGroupProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - logs:DescribeLogGroups\n          - logs:AssociateKmsKey\n          - logs:DisassociateKmsKey\n          - logs:PutRetentionPolicy\n          - logs:DeleteRetentionPolicy\n          - logs:TagLogGroup\n          - logs:UntagLogGroup\n          - logs:GetDataProtectionPolicy\n          - logs:PutDataProtectionPolicy\n          - logs:CreateLogDelivery\n          - s3:REST.PUT.OBJECT\n          - firehose:TagDeliveryStream\n        '
        raise NotImplementedError