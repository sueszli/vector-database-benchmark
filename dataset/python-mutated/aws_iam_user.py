from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class IAMUserProperties(TypedDict):
    Arn: Optional[str]
    Groups: Optional[list[str]]
    Id: Optional[str]
    LoginProfile: Optional[LoginProfile]
    ManagedPolicyArns: Optional[list[str]]
    Path: Optional[str]
    PermissionsBoundary: Optional[str]
    Policies: Optional[list[Policy]]
    Tags: Optional[list[Tag]]
    UserName: Optional[str]

class Policy(TypedDict):
    PolicyDocument: Optional[dict]
    PolicyName: Optional[str]

class LoginProfile(TypedDict):
    Password: Optional[str]
    PasswordResetRequired: Optional[bool]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class IAMUserProvider(ResourceProvider[IAMUserProperties]):
    TYPE = 'AWS::IAM::User'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[IAMUserProperties]) -> ProgressEvent[IAMUserProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Create-only properties:\n          - /properties/UserName\n\n        Read-only properties:\n          - /properties/Id\n          - /properties/Arn\n        '
        model = request.desired_state
        iam_client = request.aws_client_factory.iam
        if not request.custom_context.get(REPEATED_INVOCATION):
            if not model.get('UserName'):
                model['UserName'] = util.generate_default_name(request.stack_name, request.logical_resource_id)
            iam_client.create_user(**util.select_attributes(model, ['UserName', 'Path', 'PermissionsBoundary', 'Tags']))
            for group in model.get('Groups', []):
                iam_client.add_user_to_group(GroupName=group, UserName=model['UserName'])
            for policy_arn in model.get('ManagedPolicyArns', []):
                iam_client.attach_user_policy(UserName=model['UserName'], PolicyArn=policy_arn)
            request.custom_context[REPEATED_INVOCATION] = True
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        get_response = iam_client.get_user(UserName=model['UserName'])
        model['Id'] = get_response['User']['UserName']
        model['Arn'] = get_response['User']['Arn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[IAMUserProperties]) -> ProgressEvent[IAMUserProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[IAMUserProperties]) -> ProgressEvent[IAMUserProperties]:
        if False:
            return 10
        '\n        Delete a resource\n        '
        iam_client = request.aws_client_factory.iam
        iam_client.delete_user(UserName=request.desired_state['Id'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=None)

    def update(self, request: ResourceRequest[IAMUserProperties]) -> ProgressEvent[IAMUserProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n        '
        raise NotImplementedError