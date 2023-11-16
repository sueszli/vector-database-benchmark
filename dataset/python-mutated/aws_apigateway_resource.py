from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class ApiGatewayResourceProperties(TypedDict):
    ParentId: Optional[str]
    PathPart: Optional[str]
    RestApiId: Optional[str]
    ResourceId: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayResourceProvider(ResourceProvider[ApiGatewayResourceProperties]):
    TYPE = 'AWS::ApiGateway::Resource'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayResourceProperties]) -> ProgressEvent[ApiGatewayResourceProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/RestApiId\n          - /properties/ResourceId\n\n        Required properties:\n          - ParentId\n          - PathPart\n          - RestApiId\n\n        Create-only properties:\n          - /properties/PathPart\n          - /properties/ParentId\n          - /properties/RestApiId\n\n        Read-only properties:\n          - /properties/ResourceId\n\n        IAM permissions required:\n          - apigateway:POST\n\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        params = {'restApiId': model.get('RestApiId'), 'pathPart': model.get('PathPart'), 'parentId': model.get('ParentId')}
        if not params.get('parentId'):
            resources = apigw.get_resources(restApiId=params['restApiId'])['items']
            root_resource = ([r for r in resources if r['path'] == '/'] or [None])[0]
            if not root_resource:
                raise Exception('Unable to find root resource for REST API %s' % params['restApiId'])
            params['parentId'] = root_resource['id']
        response = apigw.create_resource(**params)
        model['ResourceId'] = response['id']
        model['ParentId'] = response['parentId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayResourceProperties]) -> ProgressEvent[ApiGatewayResourceProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayResourceProperties]) -> ProgressEvent[ApiGatewayResourceProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:DELETE\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        try:
            apigw.delete_resource(restApiId=model['RestApiId'], resourceId=model['ResourceId'])
        except apigw.exceptions.NotFoundException:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayResourceProperties]) -> ProgressEvent[ApiGatewayResourceProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:GET\n          - apigateway:PATCH\n        '
        raise NotImplementedError