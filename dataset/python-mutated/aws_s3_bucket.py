from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, TypedDict
from botocore.exceptions import ClientError
import localstack.services.cloudformation.provider_utils as util
from localstack.config import S3_STATIC_WEBSITE_HOSTNAME, S3_VIRTUAL_HOSTNAME
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.services.s3.utils import normalize_bucket_name
from localstack.utils.aws import arns
from localstack.utils.testutil import delete_all_s3_objects
from localstack.utils.urls import localstack_host

class S3BucketProperties(TypedDict):
    AccelerateConfiguration: Optional[AccelerateConfiguration]
    AccessControl: Optional[str]
    AnalyticsConfigurations: Optional[list[AnalyticsConfiguration]]
    Arn: Optional[str]
    BucketEncryption: Optional[BucketEncryption]
    BucketName: Optional[str]
    CorsConfiguration: Optional[CorsConfiguration]
    DomainName: Optional[str]
    DualStackDomainName: Optional[str]
    IntelligentTieringConfigurations: Optional[list[IntelligentTieringConfiguration]]
    InventoryConfigurations: Optional[list[InventoryConfiguration]]
    LifecycleConfiguration: Optional[LifecycleConfiguration]
    LoggingConfiguration: Optional[LoggingConfiguration]
    MetricsConfigurations: Optional[list[MetricsConfiguration]]
    NotificationConfiguration: Optional[NotificationConfiguration]
    ObjectLockConfiguration: Optional[ObjectLockConfiguration]
    ObjectLockEnabled: Optional[bool]
    OwnershipControls: Optional[OwnershipControls]
    PublicAccessBlockConfiguration: Optional[PublicAccessBlockConfiguration]
    RegionalDomainName: Optional[str]
    ReplicationConfiguration: Optional[ReplicationConfiguration]
    Tags: Optional[list[Tag]]
    VersioningConfiguration: Optional[VersioningConfiguration]
    WebsiteConfiguration: Optional[WebsiteConfiguration]
    WebsiteURL: Optional[str]

class AccelerateConfiguration(TypedDict):
    AccelerationStatus: Optional[str]

class TagFilter(TypedDict):
    Key: Optional[str]
    Value: Optional[str]

class Destination(TypedDict):
    BucketArn: Optional[str]
    Format: Optional[str]
    BucketAccountId: Optional[str]
    Prefix: Optional[str]

class DataExport(TypedDict):
    Destination: Optional[Destination]
    OutputSchemaVersion: Optional[str]

class StorageClassAnalysis(TypedDict):
    DataExport: Optional[DataExport]

class AnalyticsConfiguration(TypedDict):
    Id: Optional[str]
    StorageClassAnalysis: Optional[StorageClassAnalysis]
    Prefix: Optional[str]
    TagFilters: Optional[list[TagFilter]]

class ServerSideEncryptionByDefault(TypedDict):
    SSEAlgorithm: Optional[str]
    KMSMasterKeyID: Optional[str]

class ServerSideEncryptionRule(TypedDict):
    BucketKeyEnabled: Optional[bool]
    ServerSideEncryptionByDefault: Optional[ServerSideEncryptionByDefault]

class BucketEncryption(TypedDict):
    ServerSideEncryptionConfiguration: Optional[list[ServerSideEncryptionRule]]

class CorsRule(TypedDict):
    AllowedMethods: Optional[list[str]]
    AllowedOrigins: Optional[list[str]]
    AllowedHeaders: Optional[list[str]]
    ExposedHeaders: Optional[list[str]]
    Id: Optional[str]
    MaxAge: Optional[int]

class CorsConfiguration(TypedDict):
    CorsRules: Optional[list[CorsRule]]

class Tiering(TypedDict):
    AccessTier: Optional[str]
    Days: Optional[int]

class IntelligentTieringConfiguration(TypedDict):
    Id: Optional[str]
    Status: Optional[str]
    Tierings: Optional[list[Tiering]]
    Prefix: Optional[str]
    TagFilters: Optional[list[TagFilter]]

class InventoryConfiguration(TypedDict):
    Destination: Optional[Destination]
    Enabled: Optional[bool]
    Id: Optional[str]
    IncludedObjectVersions: Optional[str]
    ScheduleFrequency: Optional[str]
    OptionalFields: Optional[list[str]]
    Prefix: Optional[str]

class AbortIncompleteMultipartUpload(TypedDict):
    DaysAfterInitiation: Optional[int]

class NoncurrentVersionExpiration(TypedDict):
    NoncurrentDays: Optional[int]
    NewerNoncurrentVersions: Optional[int]

class NoncurrentVersionTransition(TypedDict):
    StorageClass: Optional[str]
    TransitionInDays: Optional[int]
    NewerNoncurrentVersions: Optional[int]

class Transition(TypedDict):
    StorageClass: Optional[str]
    TransitionDate: Optional[str]
    TransitionInDays: Optional[int]

class Rule(TypedDict):
    Status: Optional[str]
    AbortIncompleteMultipartUpload: Optional[AbortIncompleteMultipartUpload]
    ExpirationDate: Optional[str]
    ExpirationInDays: Optional[int]
    ExpiredObjectDeleteMarker: Optional[bool]
    Id: Optional[str]
    NoncurrentVersionExpiration: Optional[NoncurrentVersionExpiration]
    NoncurrentVersionExpirationInDays: Optional[int]
    NoncurrentVersionTransition: Optional[NoncurrentVersionTransition]
    NoncurrentVersionTransitions: Optional[list[NoncurrentVersionTransition]]
    ObjectSizeGreaterThan: Optional[str]
    ObjectSizeLessThan: Optional[str]
    Prefix: Optional[str]
    TagFilters: Optional[list[TagFilter]]
    Transition: Optional[Transition]
    Transitions: Optional[list[Transition]]

class LifecycleConfiguration(TypedDict):
    Rules: Optional[list[Rule]]

class LoggingConfiguration(TypedDict):
    DestinationBucketName: Optional[str]
    LogFilePrefix: Optional[str]

class MetricsConfiguration(TypedDict):
    Id: Optional[str]
    AccessPointArn: Optional[str]
    Prefix: Optional[str]
    TagFilters: Optional[list[TagFilter]]

class EventBridgeConfiguration(TypedDict):
    EventBridgeEnabled: Optional[bool]

class FilterRule(TypedDict):
    Name: Optional[str]
    Value: Optional[str]

class S3KeyFilter(TypedDict):
    Rules: Optional[list[FilterRule]]

class NotificationFilter(TypedDict):
    S3Key: Optional[S3KeyFilter]

class LambdaConfiguration(TypedDict):
    Event: Optional[str]
    Function: Optional[str]
    Filter: Optional[NotificationFilter]

class QueueConfiguration(TypedDict):
    Event: Optional[str]
    Queue: Optional[str]
    Filter: Optional[NotificationFilter]

class TopicConfiguration(TypedDict):
    Event: Optional[str]
    Topic: Optional[str]
    Filter: Optional[NotificationFilter]

class NotificationConfiguration(TypedDict):
    EventBridgeConfiguration: Optional[EventBridgeConfiguration]
    LambdaConfigurations: Optional[list[LambdaConfiguration]]
    QueueConfigurations: Optional[list[QueueConfiguration]]
    TopicConfigurations: Optional[list[TopicConfiguration]]

class DefaultRetention(TypedDict):
    Days: Optional[int]
    Mode: Optional[str]
    Years: Optional[int]

class ObjectLockRule(TypedDict):
    DefaultRetention: Optional[DefaultRetention]

class ObjectLockConfiguration(TypedDict):
    ObjectLockEnabled: Optional[str]
    Rule: Optional[ObjectLockRule]

class OwnershipControlsRule(TypedDict):
    ObjectOwnership: Optional[str]

class OwnershipControls(TypedDict):
    Rules: Optional[list[OwnershipControlsRule]]

class PublicAccessBlockConfiguration(TypedDict):
    BlockPublicAcls: Optional[bool]
    BlockPublicPolicy: Optional[bool]
    IgnorePublicAcls: Optional[bool]
    RestrictPublicBuckets: Optional[bool]

class DeleteMarkerReplication(TypedDict):
    Status: Optional[str]

class AccessControlTranslation(TypedDict):
    Owner: Optional[str]

class EncryptionConfiguration(TypedDict):
    ReplicaKmsKeyID: Optional[str]

class ReplicationTimeValue(TypedDict):
    Minutes: Optional[int]

class Metrics(TypedDict):
    Status: Optional[str]
    EventThreshold: Optional[ReplicationTimeValue]

class ReplicationTime(TypedDict):
    Status: Optional[str]
    Time: Optional[ReplicationTimeValue]

class ReplicationDestination(TypedDict):
    Bucket: Optional[str]
    AccessControlTranslation: Optional[AccessControlTranslation]
    Account: Optional[str]
    EncryptionConfiguration: Optional[EncryptionConfiguration]
    Metrics: Optional[Metrics]
    ReplicationTime: Optional[ReplicationTime]
    StorageClass: Optional[str]

class ReplicationRuleAndOperator(TypedDict):
    Prefix: Optional[str]
    TagFilters: Optional[list[TagFilter]]

class ReplicationRuleFilter(TypedDict):
    And: Optional[ReplicationRuleAndOperator]
    Prefix: Optional[str]
    TagFilter: Optional[TagFilter]

class ReplicaModifications(TypedDict):
    Status: Optional[str]

class SseKmsEncryptedObjects(TypedDict):
    Status: Optional[str]

class SourceSelectionCriteria(TypedDict):
    ReplicaModifications: Optional[ReplicaModifications]
    SseKmsEncryptedObjects: Optional[SseKmsEncryptedObjects]

class ReplicationRule(TypedDict):
    Destination: Optional[ReplicationDestination]
    Status: Optional[str]
    DeleteMarkerReplication: Optional[DeleteMarkerReplication]
    Filter: Optional[ReplicationRuleFilter]
    Id: Optional[str]
    Prefix: Optional[str]
    Priority: Optional[int]
    SourceSelectionCriteria: Optional[SourceSelectionCriteria]

class ReplicationConfiguration(TypedDict):
    Role: Optional[str]
    Rules: Optional[list[ReplicationRule]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]

class VersioningConfiguration(TypedDict):
    Status: Optional[str]

class RedirectRule(TypedDict):
    HostName: Optional[str]
    HttpRedirectCode: Optional[str]
    Protocol: Optional[str]
    ReplaceKeyPrefixWith: Optional[str]
    ReplaceKeyWith: Optional[str]

class RoutingRuleCondition(TypedDict):
    HttpErrorCodeReturnedEquals: Optional[str]
    KeyPrefixEquals: Optional[str]

class RoutingRule(TypedDict):
    RedirectRule: Optional[RedirectRule]
    RoutingRuleCondition: Optional[RoutingRuleCondition]

class RedirectAllRequestsTo(TypedDict):
    HostName: Optional[str]
    Protocol: Optional[str]

class WebsiteConfiguration(TypedDict):
    ErrorDocument: Optional[str]
    IndexDocument: Optional[str]
    RedirectAllRequestsTo: Optional[RedirectAllRequestsTo]
    RoutingRules: Optional[list[RoutingRule]]
REPEATED_INVOCATION = 'repeated_invocation'

class S3BucketProvider(ResourceProvider[S3BucketProperties]):
    TYPE = 'AWS::S3::Bucket'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[S3BucketProperties]) -> ProgressEvent[S3BucketProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/BucketName\n\n\n        Create-only properties:\n          - /properties/BucketName\n          - /properties/ObjectLockEnabled\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/DomainName\n          - /properties/DualStackDomainName\n          - /properties/RegionalDomainName\n          - /properties/WebsiteURL\n\n        IAM permissions required:\n          - s3:CreateBucket\n          - s3:PutBucketTagging\n          - s3:PutAnalyticsConfiguration\n          - s3:PutEncryptionConfiguration\n          - s3:PutBucketCORS\n          - s3:PutInventoryConfiguration\n          - s3:PutLifecycleConfiguration\n          - s3:PutMetricsConfiguration\n          - s3:PutBucketNotification\n          - s3:PutBucketReplication\n          - s3:PutBucketWebsite\n          - s3:PutAccelerateConfiguration\n          - s3:PutBucketPublicAccessBlock\n          - s3:PutReplicationConfiguration\n          - s3:PutObjectAcl\n          - s3:PutBucketObjectLockConfiguration\n          - s3:GetBucketAcl\n          - s3:ListBucket\n          - iam:PassRole\n          - s3:DeleteObject\n          - s3:PutBucketLogging\n          - s3:PutBucketVersioning\n          - s3:PutObjectLockConfiguration\n          - s3:PutBucketOwnershipControls\n          - s3:PutBucketIntelligentTieringConfiguration\n\n        '
        model = request.desired_state
        s3_client = request.aws_client_factory.s3
        if not model.get('BucketName'):
            model['BucketName'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        model['BucketName'] = normalize_bucket_name(model['BucketName'])
        self._create_bucket_if_does_not_exist(model, request.region_name, s3_client)
        self._setup_post_creation_attributes(model)
        if (put_config := self._get_s3_bucket_notification_config(model)):
            s3_client.put_bucket_notification_configuration(**put_config)
        if (version_conf := model.get('VersioningConfiguration')):
            s3_client.put_bucket_versioning(Bucket=model['BucketName'], VersioningConfiguration={'Status': version_conf.get('Status', 'Disabled')})
        if (cors_configuration := self._transform_cfn_cors(model.get('CorsConfiguration'))):
            s3_client.put_bucket_cors(Bucket=model['BucketName'], CORSConfiguration=cors_configuration)
        if (tags := model.get('Tags')):
            s3_client.put_bucket_tagging(Bucket=model['BucketName'], Tagging={'TagSet': tags})
        if (website_config := self._transform_website_configuration(model.get('WebsiteConfiguration'))):
            s3_client.put_bucket_website(Bucket=model['BucketName'], WebsiteConfiguration=website_config)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def _transform_website_configuration(self, website_configuration: dict) -> dict:
        if False:
            return 10
        if not website_configuration:
            return {}
        output = {}
        if (index := website_configuration.get('IndexDocument')):
            output['IndexDocument'] = {'Suffix': index}
        if (error := website_configuration.get('ErrorDocument')):
            output['ErrorDocument'] = {'Key': error}
        if (redirect_all := website_configuration.get('RedirectAllRequestsTo')):
            output['RedirectAllRequestsTo'] = redirect_all
        for r in website_configuration.get('RoutingRules', []):
            rule = {}
            if (condition := r.get('RoutingRuleCondition')):
                rule['Condition'] = condition
            if (redirect := r.get('RedirectRule')):
                rule['Redirect'] = redirect
            output.setdefault('RoutingRules', []).append(rule)
        return output

    def _transform_cfn_cors(self, cors_config):
        if False:
            for i in range(10):
                print('nop')
        if not cors_config:
            return {}
        cors_rules = []
        for cfn_rule in cors_config.get('CorsRules', []):
            rule = {'AllowedOrigins': cfn_rule.get('AllowedOrigins'), 'AllowedMethods': cfn_rule.get('AllowedMethods')}
            if (allowed_headers := cfn_rule.get('AllowedHeaders')) is not None:
                rule['AllowedHeaders'] = allowed_headers
            if (allowed_headers := cfn_rule.get('ExposedHeaders')) is not None:
                rule['ExposeHeaders'] = allowed_headers
            if (allowed_headers := cfn_rule.get('MaxAge')) is not None:
                rule['MaxAgeSeconds'] = allowed_headers
            if (allowed_headers := cfn_rule.get('Id')) is not None:
                rule['ID'] = allowed_headers
            cors_rules.append(rule)
        return {'CORSRules': cors_rules}

    def _get_s3_bucket_notification_config(self, properties: dict) -> dict | None:
        if False:
            return 10
        notif_config = properties.get('NotificationConfiguration')
        if not notif_config:
            return None
        lambda_configs = []
        queue_configs = []
        topic_configs = []
        attr_tuples = (('LambdaConfigurations', lambda_configs, 'LambdaFunctionArn', 'Function'), ('QueueConfigurations', queue_configs, 'QueueArn', 'Queue'), ('TopicConfigurations', topic_configs, 'TopicArn', 'Topic'))
        for attrs in attr_tuples:
            for notif_cfg in notif_config.get(attrs[0]) or []:
                filter_rules = notif_cfg.get('Filter', {}).get('S3Key', {}).get('Rules')
                entry = {attrs[2]: notif_cfg[attrs[3]], 'Events': [notif_cfg['Event']]}
                if filter_rules:
                    entry['Filter'] = {'Key': {'FilterRules': filter_rules}}
                attrs[1].append(entry)
        result = {'Bucket': properties.get('BucketName'), 'NotificationConfiguration': {'LambdaFunctionConfigurations': lambda_configs, 'QueueConfigurations': queue_configs, 'TopicConfigurations': topic_configs}}
        if notif_config.get('EventBridgeConfiguration', {}).get('EventBridgeEnabled'):
            result['NotificationConfiguration']['EventBridgeConfiguration'] = {}
        return result

    def _setup_post_creation_attributes(self, model):
        if False:
            for i in range(10):
                print('nop')
        model['Arn'] = arns.s3_bucket_arn(model['BucketName'])
        domain_name = f"{model['BucketName']}.{S3_VIRTUAL_HOSTNAME}"
        model['DomainName'] = domain_name
        model['RegionalDomainName'] = domain_name
        model['WebsiteURL'] = f"http://{model['BucketName']}.{S3_STATIC_WEBSITE_HOSTNAME}:{localstack_host().port}"

    def _create_bucket_if_does_not_exist(self, model, region_name, s3_client):
        if False:
            return 10
        try:
            s3_client.head_bucket(Bucket=model['BucketName'])
        except ClientError as e:
            if e.response['Error']['Message'] != 'Not Found':
                return
            params = {'Bucket': model['BucketName'], 'ACL': self._convert_acl_cf_to_s3(model.get('AccessControl', 'PublicRead'))}
            if region_name != 'us-east-1':
                params['CreateBucketConfiguration'] = {'LocationConstraint': region_name}
            s3_client.create_bucket(**params)

    def _convert_acl_cf_to_s3(self, acl):
        if False:
            print('Hello World!')
        "Convert a CloudFormation ACL string (e.g., 'PublicRead') to an S3 ACL string (e.g., 'public-read')"
        return re.sub('(?<!^)(?=[A-Z])', '-', acl).lower()

    def read(self, request: ResourceRequest[S3BucketProperties]) -> ProgressEvent[S3BucketProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - s3:GetAccelerateConfiguration\n          - s3:GetLifecycleConfiguration\n          - s3:GetBucketPublicAccessBlock\n          - s3:GetAnalyticsConfiguration\n          - s3:GetBucketCORS\n          - s3:GetEncryptionConfiguration\n          - s3:GetInventoryConfiguration\n          - s3:GetBucketLogging\n          - s3:GetMetricsConfiguration\n          - s3:GetBucketNotification\n          - s3:GetBucketVersioning\n          - s3:GetReplicationConfiguration\n          - S3:GetBucketWebsite\n          - s3:GetBucketPublicAccessBlock\n          - s3:GetBucketObjectLockConfiguration\n          - s3:GetBucketTagging\n          - s3:GetBucketOwnershipControls\n          - s3:GetIntelligentTieringConfiguration\n          - s3:ListBucket\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[S3BucketProperties]) -> ProgressEvent[S3BucketProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n        IAM permissions required:\n          - s3:DeleteBucket\n        '
        model = request.desired_state
        s3_client = request.aws_client_factory.s3
        try:
            delete_all_s3_objects(s3_client, model['BucketName'])
        except s3_client.exceptions.ClientError as e:
            if 'NoSuchBucket' not in str(e):
                raise
        s3_client.delete_bucket(Bucket=model['BucketName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[S3BucketProperties]) -> ProgressEvent[S3BucketProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n        IAM permissions required:\n          - s3:PutBucketAcl\n          - s3:PutBucketTagging\n          - s3:PutAnalyticsConfiguration\n          - s3:PutEncryptionConfiguration\n          - s3:PutBucketCORS\n          - s3:PutInventoryConfiguration\n          - s3:PutLifecycleConfiguration\n          - s3:PutMetricsConfiguration\n          - s3:PutBucketNotification\n          - s3:PutBucketReplication\n          - s3:PutBucketWebsite\n          - s3:PutAccelerateConfiguration\n          - s3:PutBucketPublicAccessBlock\n          - s3:PutReplicationConfiguration\n          - s3:PutBucketOwnershipControls\n          - s3:PutBucketIntelligentTieringConfiguration\n          - s3:DeleteBucketWebsite\n          - s3:PutBucketLogging\n          - s3:PutBucketVersioning\n          - s3:PutObjectLockConfiguration\n          - s3:DeleteBucketAnalyticsConfiguration\n          - s3:DeleteBucketCors\n          - s3:DeleteBucketMetricsConfiguration\n          - s3:DeleteBucketEncryption\n          - s3:DeleteBucketLifecycle\n          - s3:DeleteBucketReplication\n          - iam:PassRole\n        '
        raise NotImplementedError