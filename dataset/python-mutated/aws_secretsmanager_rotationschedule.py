from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class SecretsManagerRotationScheduleProperties(TypedDict):
    SecretId: Optional[str]
    HostedRotationLambda: Optional[HostedRotationLambda]
    Id: Optional[str]
    RotateImmediatelyOnUpdate: Optional[bool]
    RotationLambdaARN: Optional[str]
    RotationRules: Optional[RotationRules]

class RotationRules(TypedDict):
    AutomaticallyAfterDays: Optional[int]
    Duration: Optional[str]
    ScheduleExpression: Optional[str]

class HostedRotationLambda(TypedDict):
    RotationType: Optional[str]
    ExcludeCharacters: Optional[str]
    KmsKeyArn: Optional[str]
    MasterSecretArn: Optional[str]
    MasterSecretKmsKeyArn: Optional[str]
    RotationLambdaName: Optional[str]
    Runtime: Optional[str]
    SuperuserSecretArn: Optional[str]
    SuperuserSecretKmsKeyArn: Optional[str]
    VpcSecurityGroupIds: Optional[str]
    VpcSubnetIds: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class SecretsManagerRotationScheduleProvider(ResourceProvider[SecretsManagerRotationScheduleProperties]):
    TYPE = 'AWS::SecretsManager::RotationSchedule'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SecretsManagerRotationScheduleProperties]) -> ProgressEvent[SecretsManagerRotationScheduleProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - SecretId\n\n        Create-only properties:\n          - /properties/SecretId\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        if not model.get('Id'):
            model['Id'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[SecretsManagerRotationScheduleProperties]) -> ProgressEvent[SecretsManagerRotationScheduleProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SecretsManagerRotationScheduleProperties]) -> ProgressEvent[SecretsManagerRotationScheduleProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[SecretsManagerRotationScheduleProperties]) -> ProgressEvent[SecretsManagerRotationScheduleProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError