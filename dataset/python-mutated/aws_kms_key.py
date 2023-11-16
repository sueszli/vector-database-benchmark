from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class KMSKeyProperties(TypedDict):
    KeyPolicy: Optional[dict | str]
    Arn: Optional[str]
    Description: Optional[str]
    EnableKeyRotation: Optional[bool]
    Enabled: Optional[bool]
    KeyId: Optional[str]
    KeySpec: Optional[str]
    KeyUsage: Optional[str]
    MultiRegion: Optional[bool]
    PendingWindowInDays: Optional[int]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class KMSKeyProvider(ResourceProvider[KMSKeyProperties]):
    TYPE = 'AWS::KMS::Key'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[KMSKeyProperties]) -> ProgressEvent[KMSKeyProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/KeyId\n\n        Required properties:\n          - KeyPolicy\n\n\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/KeyId\n\n        IAM permissions required:\n          - kms:CreateKey\n          - kms:EnableKeyRotation\n          - kms:DisableKey\n          - kms:TagResource\n\n        '
        model = request.desired_state
        kms = request.aws_client_factory.kms
        params = util.select_attributes(model, ['Description', 'KeySpec', 'KeyUsage'])
        if model.get('KeyPolicy'):
            params['Policy'] = json.dumps(model['KeyPolicy'])
        if model.get('Tags'):
            params['Tags'] = [{'TagKey': tag['Key'], 'TagValue': tag['Value']} for tag in model.get('Tags', [])]
        response = kms.create_key(**params)
        model['KeyId'] = response['KeyMetadata']['KeyId']
        model['Arn'] = response['KeyMetadata']['Arn']
        if model.get('EnableKeyRotation', False):
            kms.enable_key_rotation(KeyId=model['KeyId'])
        else:
            kms.disable_key_rotation(KeyId=model['KeyId'])
        if model.get('Enabled', True):
            kms.enable_key(KeyId=model['KeyId'])
        else:
            kms.disable_key(KeyId=model['KeyId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[KMSKeyProperties]) -> ProgressEvent[KMSKeyProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - kms:DescribeKey\n          - kms:GetKeyPolicy\n          - kms:GetKeyRotationStatus\n          - kms:ListResourceTags\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[KMSKeyProperties]) -> ProgressEvent[KMSKeyProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - kms:DescribeKey\n          - kms:ScheduleKeyDeletion\n        '
        model = request.desired_state
        kms = request.aws_client_factory.kms
        kms.schedule_key_deletion(KeyId=model['KeyId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[KMSKeyProperties]) -> ProgressEvent[KMSKeyProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - kms:DescribeKey\n          - kms:DisableKey\n          - kms:DisableKeyRotation\n          - kms:EnableKey\n          - kms:EnableKeyRotation\n          - kms:PutKeyPolicy\n          - kms:TagResource\n          - kms:UntagResource\n          - kms:UpdateKeyDescription\n        '
        raise NotImplementedError