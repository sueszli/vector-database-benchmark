from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class ApiGatewayRequestValidatorProperties(TypedDict):
    RestApiId: Optional[str]
    Name: Optional[str]
    RequestValidatorId: Optional[str]
    ValidateRequestBody: Optional[bool]
    ValidateRequestParameters: Optional[bool]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayRequestValidatorProvider(ResourceProvider[ApiGatewayRequestValidatorProperties]):
    TYPE = 'AWS::ApiGateway::RequestValidator'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayRequestValidatorProperties]) -> ProgressEvent[ApiGatewayRequestValidatorProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/RestApiId\n          - /properties/RequestValidatorId\n\n        Required properties:\n          - RestApiId\n\n        Create-only properties:\n          - /properties/Name\n          - /properties/RestApiId\n\n        Read-only properties:\n          - /properties/RequestValidatorId\n\n        IAM permissions required:\n          - apigateway:POST\n          - apigateway:GET\n\n        '
        model = request.desired_state
        api = request.aws_client_factory.apigateway
        if not model.get('Name'):
            model['Name'] = util.generate_default_name(request.stack_name, request.logical_resource_id)
        response = api.create_request_validator(name=model['Name'], restApiId=model['RestApiId'], validateRequestBody=model['ValidateRequestBody'], validateRequestParameters=model['ValidateRequestParameters'])
        model['RequestValidatorId'] = response['id']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayRequestValidatorProperties]) -> ProgressEvent[ApiGatewayRequestValidatorProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayRequestValidatorProperties]) -> ProgressEvent[ApiGatewayRequestValidatorProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:DELETE\n        '
        model = request.desired_state
        api = request.aws_client_factory.apigateway
        api.delete_request_validator(restApiId=model['RestApiId'], requestValidatorId=model['RequestValidatorId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayRequestValidatorProperties]) -> ProgressEvent[ApiGatewayRequestValidatorProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:PATCH\n        '
        raise NotImplementedError