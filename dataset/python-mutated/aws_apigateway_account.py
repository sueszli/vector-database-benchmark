from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class ApiGatewayAccountProperties(TypedDict):
    CloudWatchRoleArn: Optional[str]
    Id: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayAccountProvider(ResourceProvider[ApiGatewayAccountProperties]):
    TYPE = 'AWS::ApiGateway::Account'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayAccountProperties]) -> ProgressEvent[ApiGatewayAccountProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n\n\n\n\n        Read-only properties:\n          - /properties/Id\n\n        IAM permissions required:\n          - apigateway:PATCH\n          - iam:GetRole\n          - iam:PassRole\n\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        role_arn = model['CloudWatchRoleArn']
        apigw.update_account(patchOperations=[{'op': 'replace', 'path': '/cloudwatchRoleArn', 'value': role_arn}])
        model['Id'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayAccountProperties]) -> ProgressEvent[ApiGatewayAccountProperties]:
        if False:
            return 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayAccountProperties]) -> ProgressEvent[ApiGatewayAccountProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayAccountProperties]) -> ProgressEvent[ApiGatewayAccountProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:PATCH\n          - iam:GetRole\n          - iam:PassRole\n        '
        raise NotImplementedError