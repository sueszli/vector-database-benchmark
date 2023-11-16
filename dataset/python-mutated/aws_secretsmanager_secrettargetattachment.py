from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class SecretsManagerSecretTargetAttachmentProperties(TypedDict):
    SecretId: Optional[str]
    TargetId: Optional[str]
    TargetType: Optional[str]
    Id: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class SecretsManagerSecretTargetAttachmentProvider(ResourceProvider[SecretsManagerSecretTargetAttachmentProperties]):
    TYPE = 'AWS::SecretsManager::SecretTargetAttachment'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SecretsManagerSecretTargetAttachmentProperties]) -> ProgressEvent[SecretsManagerSecretTargetAttachmentProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - TargetType\n          - TargetId\n          - SecretId\n\n\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        if not model.get('Id'):
            model['Id'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[SecretsManagerSecretTargetAttachmentProperties]) -> ProgressEvent[SecretsManagerSecretTargetAttachmentProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SecretsManagerSecretTargetAttachmentProperties]) -> ProgressEvent[SecretsManagerSecretTargetAttachmentProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[SecretsManagerSecretTargetAttachmentProperties]) -> ProgressEvent[SecretsManagerSecretTargetAttachmentProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError