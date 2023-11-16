from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.objects import keys_to_lower

class ApiGatewayApiKeyProperties(TypedDict):
    APIKeyId: Optional[str]
    CustomerId: Optional[str]
    Description: Optional[str]
    Enabled: Optional[bool]
    GenerateDistinctId: Optional[bool]
    Name: Optional[str]
    StageKeys: Optional[list[StageKey]]
    Tags: Optional[list[Tag]]
    Value: Optional[str]

class StageKey(TypedDict):
    RestApiId: Optional[str]
    StageName: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayApiKeyProvider(ResourceProvider[ApiGatewayApiKeyProperties]):
    TYPE = 'AWS::ApiGateway::ApiKey'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayApiKeyProperties]) -> ProgressEvent[ApiGatewayApiKeyProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/APIKeyId\n\n\n        Create-only properties:\n          - /properties/GenerateDistinctId\n          - /properties/Name\n          - /properties/Value\n\n        Read-only properties:\n          - /properties/APIKeyId\n\n        IAM permissions required:\n          - apigateway:POST\n          - apigateway:GET\n\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        params = util.select_attributes(model, ['Description', 'CustomerId', 'Name', 'Value', 'Enabled', 'StageKeys'])
        params = keys_to_lower(params.copy())
        if 'enabled' in params:
            params['enabled'] = bool(params['enabled'])
        if model.get('Tags'):
            params['tags'] = {tag['Key']: tag['Value'] for tag in model['Tags']}
        response = apigw.create_api_key(**params)
        model['APIKeyId'] = response['id']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayApiKeyProperties]) -> ProgressEvent[ApiGatewayApiKeyProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayApiKeyProperties]) -> ProgressEvent[ApiGatewayApiKeyProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:DELETE\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        apigw.delete_api_key(apiKey=model['APIKeyId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayApiKeyProperties]) -> ProgressEvent[ApiGatewayApiKeyProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:GET\n          - apigateway:PATCH\n          - apigateway:PUT\n          - apigateway:DELETE\n        '
        raise NotImplementedError