from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class ApiGatewayBasePathMappingProperties(TypedDict):
    DomainName: Optional[str]
    BasePath: Optional[str]
    RestApiId: Optional[str]
    Stage: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayBasePathMappingProvider(ResourceProvider[ApiGatewayBasePathMappingProperties]):
    TYPE = 'AWS::ApiGateway::BasePathMapping'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayBasePathMappingProperties]) -> ProgressEvent[ApiGatewayBasePathMappingProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/DomainName\n          - /properties/BasePath\n\n        Required properties:\n          - DomainName\n\n        Create-only properties:\n          - /properties/DomainName\n          - /properties/BasePath\n\n\n\n        IAM permissions required:\n          - apigateway:POST\n          - apigateway:GET\n\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        params = {'domainName': model.get('DomainName'), 'restApiId': model.get('RestApiId'), **({'basePath': model.get('BasePath')} if model.get('BasePath') else {}), **({'stage': model.get('Stage')} if model.get('Stage') else {})}
        response = apigw.create_base_path_mapping(**params)
        model['RestApiId'] = response['restApiId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayBasePathMappingProperties]) -> ProgressEvent[ApiGatewayBasePathMappingProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayBasePathMappingProperties]) -> ProgressEvent[ApiGatewayBasePathMappingProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:DELETE\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        apigw.delete_base_path_mapping(domainName=model['DomainName'], basePath=model['BasePath'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayBasePathMappingProperties]) -> ProgressEvent[ApiGatewayBasePathMappingProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:GET\n          - apigateway:DELETE\n          - apigateway:PATCH\n        '
        raise NotImplementedError