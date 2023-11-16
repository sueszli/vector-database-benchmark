from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class IAMInstanceProfileProperties(TypedDict):
    Roles: Optional[list[str]]
    Arn: Optional[str]
    InstanceProfileName: Optional[str]
    Path: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class IAMInstanceProfileProvider(ResourceProvider[IAMInstanceProfileProperties]):
    TYPE = 'AWS::IAM::InstanceProfile'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[IAMInstanceProfileProperties]) -> ProgressEvent[IAMInstanceProfileProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/InstanceProfileName\n\n        Required properties:\n          - Roles\n\n        Create-only properties:\n          - /properties/InstanceProfileName\n          - /properties/Path\n\n        Read-only properties:\n          - /properties/Arn\n\n        IAM permissions required:\n          - iam:CreateInstanceProfile\n          - iam:PassRole\n          - iam:AddRoleToInstanceProfile\n          - iam:GetInstanceProfile\n\n        '
        model = request.desired_state
        iam = request.aws_client_factory.iam
        role_name = model.get('InstanceProfileName')
        if not role_name:
            role_name = util.generate_default_name(request.stack_name, request.logical_resource_id)
            model['InstanceProfileName'] = role_name
        response = iam.create_instance_profile(**util.select_attributes(model, ['InstanceProfileName', 'Path']))
        for role_name in model.get('Roles', []):
            iam.add_role_to_instance_profile(InstanceProfileName=model['InstanceProfileName'], RoleName=role_name)
        model['Arn'] = response['InstanceProfile']['Arn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[IAMInstanceProfileProperties]) -> ProgressEvent[IAMInstanceProfileProperties]:
        if False:
            return 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - iam:GetInstanceProfile\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[IAMInstanceProfileProperties]) -> ProgressEvent[IAMInstanceProfileProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - iam:GetInstanceProfile\n          - iam:RemoveRoleFromInstanceProfile\n          - iam:DeleteInstanceProfile\n        '
        iam = request.aws_client_factory.iam
        instance_profile = iam.get_instance_profile(InstanceProfileName=request.previous_state['InstanceProfileName'])
        for role in instance_profile['InstanceProfile']['Roles']:
            iam.remove_role_from_instance_profile(InstanceProfileName=request.previous_state['InstanceProfileName'], RoleName=role['RoleName'])
        iam.delete_instance_profile(InstanceProfileName=request.previous_state['InstanceProfileName'])
        return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model={})

    def update(self, request: ResourceRequest[IAMInstanceProfileProperties]) -> ProgressEvent[IAMInstanceProfileProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - iam:PassRole\n          - iam:RemoveRoleFromInstanceProfile\n          - iam:AddRoleToInstanceProfile\n          - iam:GetInstanceProfile\n        '
        raise NotImplementedError