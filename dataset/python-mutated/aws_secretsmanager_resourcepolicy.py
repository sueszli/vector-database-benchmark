from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class SecretsManagerResourcePolicyProperties(TypedDict):
    ResourcePolicy: Optional[dict]
    SecretId: Optional[str]
    BlockPublicPolicy: Optional[bool]
    Id: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class SecretsManagerResourcePolicyProvider(ResourceProvider[SecretsManagerResourcePolicyProperties]):
    TYPE = 'AWS::SecretsManager::ResourcePolicy'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SecretsManagerResourcePolicyProperties]) -> ProgressEvent[SecretsManagerResourcePolicyProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - SecretId\n          - ResourcePolicy\n\n        Create-only properties:\n          - /properties/SecretId\n\n        Read-only properties:\n          - /properties/Id\n\n        '
        model = request.desired_state
        secret_manager = request.aws_client_factory.secretsmanager
        params = {'SecretId': model['SecretId'], 'ResourcePolicy': json.dumps(model['ResourcePolicy']), 'BlockPublicPolicy': model.get('BlockPublicPolicy')}
        response = secret_manager.put_resource_policy(**params)
        model['Id'] = response['ARN']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[SecretsManagerResourcePolicyProperties]) -> ProgressEvent[SecretsManagerResourcePolicyProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SecretsManagerResourcePolicyProperties]) -> ProgressEvent[SecretsManagerResourcePolicyProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n        '
        model = request.desired_state
        secret_manager = request.aws_client_factory.secretsmanager
        response = secret_manager.delete_resource_policy(SecretId=model['SecretId'])
        model['Id'] = response['ARN']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[SecretsManagerResourcePolicyProperties]) -> ProgressEvent[SecretsManagerResourcePolicyProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError