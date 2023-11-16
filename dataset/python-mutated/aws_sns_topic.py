from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.strings import short_uid

class SNSTopicProperties(TypedDict):
    ContentBasedDeduplication: Optional[bool]
    DataProtectionPolicy: Optional[dict]
    DisplayName: Optional[str]
    FifoTopic: Optional[bool]
    KmsMasterKeyId: Optional[str]
    SignatureVersion: Optional[str]
    Subscription: Optional[list[Subscription]]
    Tags: Optional[list[Tag]]
    TopicArn: Optional[str]
    TopicName: Optional[str]
    TracingConfig: Optional[str]

class Subscription(TypedDict):
    Endpoint: Optional[str]
    Protocol: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class SNSTopicProvider(ResourceProvider[SNSTopicProperties]):
    TYPE = 'AWS::SNS::Topic'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SNSTopicProperties]) -> ProgressEvent[SNSTopicProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/TopicArn\n\n\n\n        Create-only properties:\n          - /properties/TopicName\n          - /properties/FifoTopic\n\n        Read-only properties:\n          - /properties/TopicArn\n\n        IAM permissions required:\n          - sns:CreateTopic\n          - sns:TagResource\n          - sns:Subscribe\n          - sns:GetTopicAttributes\n          - sns:PutDataProtectionPolicy\n\n        '
        model = request.desired_state
        sns = request.aws_client_factory.sns
        attributes = {k: v for (k, v) in model.items() if v is not None if k != 'TopicName'}
        if attributes.get('FifoTopic') is not None:
            attributes['FifoTopic'] = str(attributes.get('FifoTopic'))
        if attributes.get('ContentBasedDeduplication') is not None:
            attributes['ContentBasedDeduplication'] = str(attributes.get('ContentBasedDeduplication'))
        subscriptions = []
        if attributes.get('Subscription') is not None:
            subscriptions = attributes['Subscription']
            del attributes['Subscription']
        tags = []
        if attributes.get('Tags') is not None:
            tags = attributes['Tags']
            del attributes['Tags']
        if model.get('TopicName') is None:
            model['TopicName'] = f'topic-{short_uid()}'
        create_sns_response = sns.create_topic(Name=model['TopicName'], Attributes=attributes)
        request.custom_context[REPEATED_INVOCATION] = True
        model['TopicArn'] = create_sns_response['TopicArn']
        for subscription in subscriptions:
            sns.subscribe(TopicArn=model['TopicArn'], Protocol=subscription['Protocol'], Endpoint=subscription['Endpoint'])
        if tags:
            sns.tag_resource(ResourceArn=model['TopicArn'], Tags=tags)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[SNSTopicProperties]) -> ProgressEvent[SNSTopicProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - sns:GetTopicAttributes\n          - sns:ListTagsForResource\n          - sns:ListSubscriptionsByTopic\n          - sns:GetDataProtectionPolicy\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SNSTopicProperties]) -> ProgressEvent[SNSTopicProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - sns:DeleteTopic\n        '
        model = request.desired_state
        sns = request.aws_client_factory.sns
        sns.delete_topic(TopicArn=model['TopicArn'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[SNSTopicProperties]) -> ProgressEvent[SNSTopicProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - sns:SetTopicAttributes\n          - sns:TagResource\n          - sns:UntagResource\n          - sns:Subscribe\n          - sns:Unsubscribe\n          - sns:GetTopicAttributes\n          - sns:ListTagsForResource\n          - sns:ListSubscriptionsByTopic\n          - sns:GetDataProtectionPolicy\n          - sns:PutDataProtectionPolicy\n        '
        raise NotImplementedError