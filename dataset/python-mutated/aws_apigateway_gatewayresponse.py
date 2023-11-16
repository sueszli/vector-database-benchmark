from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.objects import keys_to_lower

class ApiGatewayGatewayResponseProperties(TypedDict):
    ResponseType: Optional[str]
    RestApiId: Optional[str]
    Id: Optional[str]
    ResponseParameters: Optional[dict]
    ResponseTemplates: Optional[dict]
    StatusCode: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayGatewayResponseProvider(ResourceProvider[ApiGatewayGatewayResponseProperties]):
    TYPE = 'AWS::ApiGateway::GatewayResponse'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayGatewayResponseProperties]) -> ProgressEvent[ApiGatewayGatewayResponseProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - ResponseType\n          - RestApiId\n\n        Create-only properties:\n          - /properties/ResponseType\n          - /properties/RestApiId\n\n        Read-only properties:\n          - /properties/Id\n\n        IAM permissions required:\n          - apigateway:PUT\n          - apigateway:GET\n\n        '
        model = request.desired_state
        api = request.aws_client_factory.apigateway
        model['Id'] = util.generate_default_name_without_stack(request.logical_resource_id)
        params = util.select_attributes(model, ['RestApiId', 'ResponseType', 'StatusCode', 'ResponseParameters', 'ResponseTemplates'])
        params = keys_to_lower(params.copy())
        api.put_gateway_response(**params)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayGatewayResponseProperties]) -> ProgressEvent[ApiGatewayGatewayResponseProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayGatewayResponseProperties]) -> ProgressEvent[ApiGatewayGatewayResponseProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:GET\n          - apigateway:DELETE\n        '
        model = request.desired_state
        api = request.aws_client_factory.apigateway
        api.delete_gateway_response(restApiId=model['RestApiId'], responseType=model['ResponseType'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayGatewayResponseProperties]) -> ProgressEvent[ApiGatewayGatewayResponseProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:GET\n          - apigateway:PUT\n        '
        raise NotImplementedError