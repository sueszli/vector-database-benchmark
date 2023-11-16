from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class IAMManagedPolicyProperties(TypedDict):
    PolicyDocument: Optional[dict]
    Description: Optional[str]
    Groups: Optional[list[str]]
    Id: Optional[str]
    ManagedPolicyName: Optional[str]
    Path: Optional[str]
    Roles: Optional[list[str]]
    Users: Optional[list[str]]
REPEATED_INVOCATION = 'repeated_invocation'

class IAMManagedPolicyProvider(ResourceProvider[IAMManagedPolicyProperties]):
    TYPE = 'AWS::IAM::ManagedPolicy'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[IAMManagedPolicyProperties]) -> ProgressEvent[IAMManagedPolicyProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - PolicyDocument\n\n        Create-only properties:\n          - /properties/ManagedPolicyName\n          - /properties/Description\n          - /properties/Path\n\n        Read-only properties:\n          - /properties/Id\n\n        '
        model = request.desired_state
        iam_client = request.aws_client_factory.iam
        group_name = model.get('ManagedPolicyName')
        if not group_name:
            group_name = util.generate_default_name(request.stack_name, request.logical_resource_id)
            model['ManagedPolicyName'] = group_name
        policy_doc = json.dumps(util.remove_none_values(model['PolicyDocument']))
        policy = iam_client.create_policy(PolicyName=model['ManagedPolicyName'], PolicyDocument=policy_doc)
        model['Id'] = policy['Policy']['Arn']
        policy_arn = policy['Policy']['Arn']
        for role in model.get('Roles', []):
            iam_client.attach_role_policy(RoleName=role, PolicyArn=policy_arn)
        for user in model.get('Users', []):
            iam_client.attach_user_policy(UserName=user, PolicyArn=policy_arn)
        for group in model.get('Groups', []):
            iam_client.attach_group_policy(GroupName=group, PolicyArn=policy_arn)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[IAMManagedPolicyProperties]) -> ProgressEvent[IAMManagedPolicyProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[IAMManagedPolicyProperties]) -> ProgressEvent[IAMManagedPolicyProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n        '
        iam_client = request.aws_client_factory.iam
        model = request.previous_state
        for role in model.get('Roles', []):
            iam_client.detach_role_policy(RoleName=role, PolicyArn=model['Id'])
        for user in model.get('Users', []):
            iam_client.detach_user_policy(UserName=user, PolicyArn=model['Id'])
        for group in model.get('Groups', []):
            iam_client.detach_group_policy(GroupName=group, PolicyArn=model['Id'])
        iam_client.delete_policy(PolicyArn=model['Id'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def update(self, request: ResourceRequest[IAMManagedPolicyProperties]) -> ProgressEvent[IAMManagedPolicyProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n        '
        raise NotImplementedError