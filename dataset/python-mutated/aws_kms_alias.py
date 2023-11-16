from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class KMSAliasProperties(TypedDict):
    AliasName: Optional[str]
    TargetKeyId: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class KMSAliasProvider(ResourceProvider[KMSAliasProperties]):
    TYPE = 'AWS::KMS::Alias'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[KMSAliasProperties]) -> ProgressEvent[KMSAliasProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/AliasName\n\n        Required properties:\n          - AliasName\n          - TargetKeyId\n\n        Create-only properties:\n          - /properties/AliasName\n\n\n\n        IAM permissions required:\n          - kms:CreateAlias\n\n        '
        model = request.desired_state
        kms = request.aws_client_factory.kms
        kms.create_alias(AliasName=model['AliasName'], TargetKeyId=model['TargetKeyId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[KMSAliasProperties]) -> ProgressEvent[KMSAliasProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - kms:ListAliases\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[KMSAliasProperties]) -> ProgressEvent[KMSAliasProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - kms:DeleteAlias\n        '
        model = request.desired_state
        kms = request.aws_client_factory.kms
        kms.delete_alias(AliasName=model['AliasName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[KMSAliasProperties]) -> ProgressEvent[KMSAliasProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n        IAM permissions required:\n          - kms:UpdateAlias\n        '
        raise NotImplementedError