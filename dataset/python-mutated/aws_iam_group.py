from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class IAMGroupProperties(TypedDict):
    Arn: Optional[str]
    GroupName: Optional[str]
    Id: Optional[str]
    ManagedPolicyArns: Optional[list[str]]
    Path: Optional[str]
    Policies: Optional[list[Policy]]

class Policy(TypedDict):
    PolicyDocument: Optional[dict]
    PolicyName: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class IAMGroupProvider(ResourceProvider[IAMGroupProperties]):
    TYPE = 'AWS::IAM::Group'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[IAMGroupProperties]) -> ProgressEvent[IAMGroupProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Create-only properties:\n          - /properties/GroupName\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/Id\n        '
        model = request.desired_state
        iam_client = request.aws_client_factory.iam
        group_name = model.get('GroupName')
        if not group_name:
            group_name = util.generate_default_name(request.stack_name, request.logical_resource_id)
            model['GroupName'] = group_name
        create_group_result = iam_client.create_group(**util.select_attributes(model, ['GroupName', 'Path']))
        model['Id'] = create_group_result['Group']['GroupName']
        model['Arn'] = create_group_result['Group']['Arn']
        for managed_policy in model.get('ManagedPolicyArns', []):
            iam_client.attach_group_policy(GroupName=group_name, PolicyArn=managed_policy)
        for inline_policy in model.get('Policies', []):
            doc = json.dumps(inline_policy.get('PolicyDocument'))
            iam_client.put_group_policy(GroupName=group_name, PolicyName=inline_policy.get('PolicyName'), PolicyDocument=doc)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[IAMGroupProperties]) -> ProgressEvent[IAMGroupProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[IAMGroupProperties]) -> ProgressEvent[IAMGroupProperties]:
        if False:
            return 10
        '\n        Delete a resource\n        '
        model = request.desired_state
        iam_client = request.aws_client_factory.iam
        for managed_policy in model.get('ManagedPolicyArns', []):
            iam_client.detach_group_policy(GroupName=model['GroupName'], PolicyArn=managed_policy)
        for inline_policy in model.get('Policies', []):
            iam_client.delete_group_policy(GroupName=model['GroupName'], PolicyName=inline_policy.get('PolicyName'))
        iam_client.delete_group(GroupName=model['GroupName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[IAMGroupProperties]) -> ProgressEvent[IAMGroupProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n        '
        raise NotImplementedError