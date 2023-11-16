from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class DynamoDBTableProperties(TypedDict):
    KeySchema: Optional[list[KeySchema] | dict]
    Arn: Optional[str]
    AttributeDefinitions: Optional[list[AttributeDefinition]]
    BillingMode: Optional[str]
    ContributorInsightsSpecification: Optional[ContributorInsightsSpecification]
    DeletionProtectionEnabled: Optional[bool]
    GlobalSecondaryIndexes: Optional[list[GlobalSecondaryIndex]]
    ImportSourceSpecification: Optional[ImportSourceSpecification]
    KinesisStreamSpecification: Optional[KinesisStreamSpecification]
    LocalSecondaryIndexes: Optional[list[LocalSecondaryIndex]]
    PointInTimeRecoverySpecification: Optional[PointInTimeRecoverySpecification]
    ProvisionedThroughput: Optional[ProvisionedThroughput]
    SSESpecification: Optional[SSESpecification]
    StreamArn: Optional[str]
    StreamSpecification: Optional[StreamSpecification]
    TableClass: Optional[str]
    TableName: Optional[str]
    Tags: Optional[list[Tag]]
    TimeToLiveSpecification: Optional[TimeToLiveSpecification]

class AttributeDefinition(TypedDict):
    AttributeName: Optional[str]
    AttributeType: Optional[str]

class KeySchema(TypedDict):
    AttributeName: Optional[str]
    KeyType: Optional[str]

class Projection(TypedDict):
    NonKeyAttributes: Optional[list[str]]
    ProjectionType: Optional[str]

class ProvisionedThroughput(TypedDict):
    ReadCapacityUnits: Optional[int]
    WriteCapacityUnits: Optional[int]

class ContributorInsightsSpecification(TypedDict):
    Enabled: Optional[bool]

class GlobalSecondaryIndex(TypedDict):
    IndexName: Optional[str]
    KeySchema: Optional[list[KeySchema]]
    Projection: Optional[Projection]
    ContributorInsightsSpecification: Optional[ContributorInsightsSpecification]
    ProvisionedThroughput: Optional[ProvisionedThroughput]

class LocalSecondaryIndex(TypedDict):
    IndexName: Optional[str]
    KeySchema: Optional[list[KeySchema]]
    Projection: Optional[Projection]

class PointInTimeRecoverySpecification(TypedDict):
    PointInTimeRecoveryEnabled: Optional[bool]

class SSESpecification(TypedDict):
    SSEEnabled: Optional[bool]
    KMSMasterKeyId: Optional[str]
    SSEType: Optional[str]

class StreamSpecification(TypedDict):
    StreamViewType: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]

class TimeToLiveSpecification(TypedDict):
    AttributeName: Optional[str]
    Enabled: Optional[bool]

class KinesisStreamSpecification(TypedDict):
    StreamArn: Optional[str]

class S3BucketSource(TypedDict):
    S3Bucket: Optional[str]
    S3BucketOwner: Optional[str]
    S3KeyPrefix: Optional[str]

class Csv(TypedDict):
    Delimiter: Optional[str]
    HeaderList: Optional[list[str]]

class InputFormatOptions(TypedDict):
    Csv: Optional[Csv]

class ImportSourceSpecification(TypedDict):
    InputFormat: Optional[str]
    S3BucketSource: Optional[S3BucketSource]
    InputCompressionType: Optional[str]
    InputFormatOptions: Optional[InputFormatOptions]
REPEATED_INVOCATION = 'repeated_invocation'

class DynamoDBTableProvider(ResourceProvider[DynamoDBTableProperties]):
    TYPE = 'AWS::DynamoDB::Table'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[DynamoDBTableProperties]) -> ProgressEvent[DynamoDBTableProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/TableName\n\n        Required properties:\n          - KeySchema\n\n        Create-only properties:\n          - /properties/TableName\n          - /properties/ImportSourceSpecification\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/StreamArn\n\n        IAM permissions required:\n          - dynamodb:CreateTable\n          - dynamodb:DescribeImport\n          - dynamodb:DescribeTable\n          - dynamodb:DescribeTimeToLive\n          - dynamodb:UpdateTimeToLive\n          - dynamodb:UpdateContributorInsights\n          - dynamodb:UpdateContinuousBackups\n          - dynamodb:DescribeContinuousBackups\n          - dynamodb:DescribeContributorInsights\n          - dynamodb:EnableKinesisStreamingDestination\n          - dynamodb:DisableKinesisStreamingDestination\n          - dynamodb:DescribeKinesisStreamingDestination\n          - dynamodb:ImportTable\n          - dynamodb:ListTagsOfResource\n          - dynamodb:TagResource\n          - dynamodb:UpdateTable\n          - kinesis:DescribeStream\n          - kinesis:PutRecords\n          - iam:CreateServiceLinkedRole\n          - kms:CreateGrant\n          - kms:Decrypt\n          - kms:Describe*\n          - kms:Encrypt\n          - kms:Get*\n          - kms:List*\n          - kms:RevokeGrant\n          - logs:CreateLogGroup\n          - logs:CreateLogStream\n          - logs:DescribeLogGroups\n          - logs:DescribeLogStreams\n          - logs:PutLogEvents\n          - logs:PutRetentionPolicy\n          - s3:GetObject\n          - s3:GetObjectMetadata\n          - s3:ListBucket\n\n        '
        model = request.desired_state
        if not request.custom_context.get(REPEATED_INVOCATION):
            request.custom_context[REPEATED_INVOCATION] = True
            if not model.get('TableName'):
                model['TableName'] = util.generate_default_name(request.stack_name, request.logical_resource_id)
            if model.get('ProvisionedThroughput'):
                model['ProvisionedThroughput'] = self.get_ddb_provisioned_throughput(model)
            if model.get('GlobalSecondaryIndexes'):
                model['GlobalSecondaryIndexes'] = self.get_ddb_global_sec_indexes(model)
            properties = ['TableName', 'AttributeDefinitions', 'KeySchema', 'BillingMode', 'ProvisionedThroughput', 'LocalSecondaryIndexes', 'GlobalSecondaryIndexes', 'Tags']
            create_params = util.select_attributes(model, properties)
            if (stream_spec := model.get('StreamSpecification')):
                create_params['StreamSpecification'] = {'StreamEnabled': True, **(stream_spec or {})}
            response = request.aws_client_factory.dynamodb.create_table(**create_params)
            model['Arn'] = response['TableDescription']['TableArn']
            if model.get('KinesisStreamSpecification'):
                request.aws_client_factory.dynamodb.enable_kinesis_streaming_destination(**self.get_ddb_kinesis_stream_specification(model))
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        description = request.aws_client_factory.dynamodb.describe_table(TableName=model['TableName'])
        if description['Table']['TableStatus'] != 'ACTIVE':
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        if description['Table'].get('LatestStreamArn'):
            model['StreamArn'] = description['Table']['LatestStreamArn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[DynamoDBTableProperties]) -> ProgressEvent[DynamoDBTableProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - dynamodb:DescribeTable\n          - dynamodb:DescribeContinuousBackups\n          - dynamodb:DescribeContributorInsights\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[DynamoDBTableProperties]) -> ProgressEvent[DynamoDBTableProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - dynamodb:DeleteTable\n          - dynamodb:DescribeTable\n        '
        model = request.desired_state
        if not request.custom_context.get(REPEATED_INVOCATION):
            request.custom_context[REPEATED_INVOCATION] = True
            request.aws_client_factory.dynamodb.delete_table(TableName=model['TableName'])
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        try:
            table_state = request.aws_client_factory.dynamodb.describe_table(TableName=model['TableName'])
            match table_state['Table']['TableStatus']:
                case 'DELETING':
                    return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
                case invalid_state:
                    return ProgressEvent(status=OperationStatus.FAILED, message=f"Table deletion failed. Table {model['TableName']} found in state {invalid_state}", resource_model={})
        except request.aws_client_factory.dynamodb.exceptions.TableNotFoundException:
            return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[DynamoDBTableProperties]) -> ProgressEvent[DynamoDBTableProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - dynamodb:UpdateTable\n          - dynamodb:DescribeTable\n          - dynamodb:DescribeTimeToLive\n          - dynamodb:UpdateTimeToLive\n          - dynamodb:UpdateContinuousBackups\n          - dynamodb:UpdateContributorInsights\n          - dynamodb:DescribeContinuousBackups\n          - dynamodb:DescribeKinesisStreamingDestination\n          - dynamodb:ListTagsOfResource\n          - dynamodb:TagResource\n          - dynamodb:UntagResource\n          - dynamodb:DescribeContributorInsights\n          - dynamodb:EnableKinesisStreamingDestination\n          - dynamodb:DisableKinesisStreamingDestination\n          - kinesis:DescribeStream\n          - kinesis:PutRecords\n          - iam:CreateServiceLinkedRole\n          - kms:CreateGrant\n          - kms:Describe*\n          - kms:Get*\n          - kms:List*\n          - kms:RevokeGrant\n        '
        raise NotImplementedError

    def get_ddb_provisioned_throughput(self, properties: dict) -> dict | None:
        if False:
            while True:
                i = 10
        args = properties.get('ProvisionedThroughput')
        if args == 'AWS::NoValue':
            return None
        is_ondemand = properties.get('BillingMode') == 'PAY_PER_REQUEST'
        if args is None:
            if is_ondemand:
                return
            return {'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
        if isinstance(args['ReadCapacityUnits'], str):
            args['ReadCapacityUnits'] = int(args['ReadCapacityUnits'])
        if isinstance(args['WriteCapacityUnits'], str):
            args['WriteCapacityUnits'] = int(args['WriteCapacityUnits'])
        return args

    def get_ddb_global_sec_indexes(self, properties: dict) -> list | None:
        if False:
            i = 10
            return i + 15
        args: list = properties.get('GlobalSecondaryIndexes')
        is_ondemand = properties.get('BillingMode') == 'PAY_PER_REQUEST'
        if not args:
            return
        for index in args:
            index.pop('ContributorInsightsSpecification', None)
            provisioned_throughput = index.get('ProvisionedThroughput')
            if is_ondemand and provisioned_throughput is None:
                pass
            elif provisioned_throughput is not None:
                if isinstance((read_units := provisioned_throughput['ReadCapacityUnits']), str):
                    provisioned_throughput['ReadCapacityUnits'] = int(read_units)
                if isinstance((write_units := provisioned_throughput['WriteCapacityUnits']), str):
                    provisioned_throughput['WriteCapacityUnits'] = int(write_units)
            else:
                raise Exception("Can't specify ProvisionedThroughput with PAY_PER_REQUEST")
        return args

    def get_ddb_kinesis_stream_specification(self, properties: dict) -> dict:
        if False:
            return 10
        args = properties.get('KinesisStreamSpecification')
        if args:
            args['TableName'] = properties['TableName']
        return args