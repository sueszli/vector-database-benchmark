from __future__ import annotations
import json
import random
import string
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class IAMPolicyProperties(TypedDict):
    PolicyDocument: Optional[dict]
    PolicyName: Optional[str]
    Groups: Optional[list[str]]
    Id: Optional[str]
    Roles: Optional[list[str]]
    Users: Optional[list[str]]
REPEATED_INVOCATION = 'repeated_invocation'

class IAMPolicyProvider(ResourceProvider[IAMPolicyProperties]):
    TYPE = 'AWS::IAM::Policy'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[IAMPolicyProperties]) -> ProgressEvent[IAMPolicyProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - PolicyDocument\n          - PolicyName\n\n        Read-only properties:\n          - /properties/Id\n\n        '
        model = request.desired_state
        iam_client = request.aws_client_factory.iam
        policy_doc = json.dumps(util.remove_none_values(model['PolicyDocument']))
        policy_name = model['PolicyName']
        if not any([model.get('Roles'), model.get('Users'), model.get('Groups')]):
            return ProgressEvent(status=OperationStatus.FAILED, resource_model={}, error_code='InvalidRequest', message='At least one of [Groups,Roles,Users] must be non-empty.')
        for role in model.get('Roles', []):
            iam_client.put_role_policy(RoleName=role, PolicyName=policy_name, PolicyDocument=policy_doc)
        for user in model.get('Users', []):
            iam_client.put_user_policy(UserName=user, PolicyName=policy_name, PolicyDocument=policy_doc)
        for group in model.get('Groups', []):
            iam_client.put_group_policy(GroupName=group, PolicyName=policy_name, PolicyDocument=policy_doc)
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=13))
        model['Id'] = f"stack-{model.get('PolicyName', '')[:4]}-{suffix}"
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[IAMPolicyProperties]) -> ProgressEvent[IAMPolicyProperties]:
        if False:
            return 10
        '\n        Fetch resource information\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[IAMPolicyProperties]) -> ProgressEvent[IAMPolicyProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n        '
        iam = request.aws_client_factory.iam
        model = request.previous_state
        policy_name = request.previous_state['PolicyName']
        for role in model.get('Roles', []):
            iam.delete_role_policy(RoleName=role, PolicyName=policy_name)
        for user in model.get('Users', []):
            iam.delete_user_policy(UserName=user, PolicyName=policy_name)
        for group in model.get('Groups', []):
            iam.delete_group_policy(GroupName=group, PolicyName=policy_name)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[IAMPolicyProperties]) -> ProgressEvent[IAMPolicyProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n        '
        iam_client = request.aws_client_factory.iam
        model = request.desired_state
        policy_doc = json.dumps(util.remove_none_values(model['PolicyDocument']))
        policy_name = model['PolicyName']
        for role in model.get('Roles', []):
            iam_client.put_role_policy(RoleName=role, PolicyName=policy_name, PolicyDocument=policy_doc)
        for user in model.get('Users', []):
            iam_client.put_user_policy(UserName=user, PolicyName=policy_name, PolicyDocument=policy_doc)
        for group in model.get('Groups', []):
            iam_client.put_group_policy(GroupName=group, PolicyName=policy_name, PolicyDocument=policy_doc)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={**request.previous_state, **request.desired_state})