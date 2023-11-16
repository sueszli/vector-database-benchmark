from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.objects import keys_to_lower

class ApiGatewayUsagePlanKeyProperties(TypedDict):
    KeyId: Optional[str]
    KeyType: Optional[str]
    UsagePlanId: Optional[str]
    Id: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayUsagePlanKeyProvider(ResourceProvider[ApiGatewayUsagePlanKeyProperties]):
    TYPE = 'AWS::ApiGateway::UsagePlanKey'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayUsagePlanKeyProperties]) -> ProgressEvent[ApiGatewayUsagePlanKeyProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - KeyType\n          - UsagePlanId\n          - KeyId\n\n        Create-only properties:\n          - /properties/KeyId\n          - /properties/UsagePlanId\n          - /properties/KeyType\n\n        Read-only properties:\n          - /properties/Id\n\n        IAM permissions required:\n          - apigateway:POST\n          - apigateway:GET\n\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        params = keys_to_lower(model.copy())
        result = apigw.create_usage_plan_key(**params)
        model['Id'] = result['id']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayUsagePlanKeyProperties]) -> ProgressEvent[ApiGatewayUsagePlanKeyProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayUsagePlanKeyProperties]) -> ProgressEvent[ApiGatewayUsagePlanKeyProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:DELETE\n        '
        raise NotImplementedError
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        apigw.delete_usage_plan_key(usagePlanId=model['UsagePlanId'], keyId=model['KeyId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayUsagePlanKeyProperties]) -> ProgressEvent[ApiGatewayUsagePlanKeyProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError