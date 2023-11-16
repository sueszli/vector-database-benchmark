from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.functions import call_safe

class IAMRoleProperties(TypedDict):
    AssumeRolePolicyDocument: Optional[dict | str]
    Arn: Optional[str]
    Description: Optional[str]
    ManagedPolicyArns: Optional[list[str]]
    MaxSessionDuration: Optional[int]
    Path: Optional[str]
    PermissionsBoundary: Optional[str]
    Policies: Optional[list[Policy]]
    RoleId: Optional[str]
    RoleName: Optional[str]
    Tags: Optional[list[Tag]]

class Policy(TypedDict):
    PolicyDocument: Optional[str | dict]
    PolicyName: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'
IAM_POLICY_VERSION = '2012-10-17'

class IAMRoleProvider(ResourceProvider[IAMRoleProperties]):
    TYPE = 'AWS::IAM::Role'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[IAMRoleProperties]) -> ProgressEvent[IAMRoleProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/RoleName\n\n        Required properties:\n          - AssumeRolePolicyDocument\n\n        Create-only properties:\n          - /properties/Path\n          - /properties/RoleName\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/RoleId\n\n        IAM permissions required:\n          - iam:CreateRole\n          - iam:PutRolePolicy\n          - iam:AttachRolePolicy\n          - iam:GetRolePolicy <- not in use right now\n\n        '
        model = request.desired_state
        iam = request.aws_client_factory.iam
        role_name = model.get('RoleName')
        if not role_name:
            role_name = util.generate_default_name(request.stack_name, request.logical_resource_id)
            model['RoleName'] = role_name
        create_role_response = iam.create_role(**{k: v for (k, v) in model.items() if k not in ['ManagedPolicyArns', 'Policies', 'AssumeRolePolicyDocument']}, AssumeRolePolicyDocument=json.dumps(model['AssumeRolePolicyDocument']))
        policy_arns = model.get('ManagedPolicyArns', [])
        for arn in policy_arns:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=arn)
        inline_policies = model.get('Policies', [])
        for policy in inline_policies:
            if not isinstance(policy, dict):
                request.logger.info('Invalid format of policy for IAM role "%s": %s', model.get('RoleName'), policy)
                continue
            pol_name = policy.get('PolicyName')
            doc = dict(policy['PolicyDocument'])
            doc = util.remove_none_values(doc)
            doc['Version'] = doc.get('Version') or IAM_POLICY_VERSION
            statements = doc['Statement']
            statements = statements if isinstance(statements, list) else [statements]
            for statement in statements:
                if isinstance(statement.get('Resource'), list):
                    statement['Resource'] = [r for r in statement['Resource'] if r]
            doc = json.dumps(doc)
            iam.put_role_policy(RoleName=model['RoleName'], PolicyName=pol_name, PolicyDocument=doc)
        model['Arn'] = create_role_response['Role']['Arn']
        model['RoleId'] = create_role_response['Role']['RoleId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[IAMRoleProperties]) -> ProgressEvent[IAMRoleProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - iam:GetRole\n          - iam:ListAttachedRolePolicies\n          - iam:ListRolePolicies\n          - iam:GetRolePolicy\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[IAMRoleProperties]) -> ProgressEvent[IAMRoleProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - iam:DeleteRole\n          - iam:DetachRolePolicy\n          - iam:DeleteRolePolicy\n          - iam:GetRole\n          - iam:ListAttachedRolePolicies\n          - iam:ListRolePolicies\n        '
        iam_client = request.aws_client_factory.iam
        role_name = request.previous_state['RoleName']
        for policy in iam_client.list_attached_role_policies(RoleName=role_name).get('AttachedPolicies', []):
            call_safe(iam_client.detach_role_policy, kwargs={'RoleName': role_name, 'PolicyArn': policy['PolicyArn']})
        for inline_policy_name in iam_client.list_role_policies(RoleName=role_name).get('PolicyNames', []):
            call_safe(iam_client.delete_role_policy, kwargs={'RoleName': role_name, 'PolicyName': inline_policy_name})
        iam_client.delete_role(RoleName=role_name)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[IAMRoleProperties]) -> ProgressEvent[IAMRoleProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - iam:UpdateRole\n          - iam:UpdateRoleDescription\n          - iam:UpdateAssumeRolePolicy\n          - iam:DetachRolePolicy\n          - iam:AttachRolePolicy\n          - iam:DeleteRolePermissionsBoundary\n          - iam:PutRolePermissionsBoundary\n          - iam:DeleteRolePolicy\n          - iam:PutRolePolicy\n          - iam:TagRole\n          - iam:UntagRole\n        '
        props = request.desired_state
        _states = request.previous_state
        props_policy = props.get('AssumeRolePolicyDocument')
        new_role_name = props.get('RoleName')
        name_changed = new_role_name and new_role_name != _states['RoleName']
        policy_changed = props_policy and props_policy != _states.get('AssumeRolePolicyDocument', '')
        managed_policy_arns_changed = props.get('ManagedPolicyArns', []) != _states.get('ManagedPolicyArns', [])
        if name_changed or policy_changed or managed_policy_arns_changed:
            self.delete(request)
            return self.create(request)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=request.previous_state)