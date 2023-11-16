from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class ApiGatewayModelProperties(TypedDict):
    RestApiId: Optional[str]
    ContentType: Optional[str]
    Description: Optional[str]
    Name: Optional[str]
    Schema: Optional[dict | str]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayModelProvider(ResourceProvider[ApiGatewayModelProperties]):
    TYPE = 'AWS::ApiGateway::Model'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayModelProperties]) -> ProgressEvent[ApiGatewayModelProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/RestApiId\n          - /properties/Name\n\n        Required properties:\n          - RestApiId\n\n        Create-only properties:\n          - /properties/ContentType\n          - /properties/Name\n          - /properties/RestApiId\n\n\n\n        IAM permissions required:\n          - apigateway:POST\n          - apigateway:GET\n\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        if not model.get('Name'):
            model['Name'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        if not model.get('ContentType'):
            model['ContentType'] = 'application/json'
        schema = json.dumps(model.get('Schema', {}))
        apigw.create_model(restApiId=model['RestApiId'], name=model['Name'], contentType=model['ContentType'], schema=schema)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayModelProperties]) -> ProgressEvent[ApiGatewayModelProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayModelProperties]) -> ProgressEvent[ApiGatewayModelProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:GET\n          - apigateway:DELETE\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        try:
            apigw.delete_model(modelName=model['Name'], restApiId=model['RestApiId'])
        except apigw.exceptions.NotFoundException:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayModelProperties]) -> ProgressEvent[ApiGatewayModelProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:PATCH\n          - apigateway:GET\n        '
        raise NotImplementedError