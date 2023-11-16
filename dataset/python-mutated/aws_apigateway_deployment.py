from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class ApiGatewayDeploymentProperties(TypedDict):
    RestApiId: Optional[str]
    DeploymentCanarySettings: Optional[DeploymentCanarySettings]
    DeploymentId: Optional[str]
    Description: Optional[str]
    StageDescription: Optional[StageDescription]
    StageName: Optional[str]

class DeploymentCanarySettings(TypedDict):
    PercentTraffic: Optional[float]
    StageVariableOverrides: Optional[dict]
    UseStageCache: Optional[bool]

class AccessLogSetting(TypedDict):
    DestinationArn: Optional[str]
    Format: Optional[str]

class CanarySetting(TypedDict):
    PercentTraffic: Optional[float]
    StageVariableOverrides: Optional[dict]
    UseStageCache: Optional[bool]

class MethodSetting(TypedDict):
    CacheDataEncrypted: Optional[bool]
    CacheTtlInSeconds: Optional[int]
    CachingEnabled: Optional[bool]
    DataTraceEnabled: Optional[bool]
    HttpMethod: Optional[str]
    LoggingLevel: Optional[str]
    MetricsEnabled: Optional[bool]
    ResourcePath: Optional[str]
    ThrottlingBurstLimit: Optional[int]
    ThrottlingRateLimit: Optional[float]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]

class StageDescription(TypedDict):
    AccessLogSetting: Optional[AccessLogSetting]
    CacheClusterEnabled: Optional[bool]
    CacheClusterSize: Optional[str]
    CacheDataEncrypted: Optional[bool]
    CacheTtlInSeconds: Optional[int]
    CachingEnabled: Optional[bool]
    CanarySetting: Optional[CanarySetting]
    ClientCertificateId: Optional[str]
    DataTraceEnabled: Optional[bool]
    Description: Optional[str]
    DocumentationVersion: Optional[str]
    LoggingLevel: Optional[str]
    MethodSettings: Optional[list[MethodSetting]]
    MetricsEnabled: Optional[bool]
    Tags: Optional[list[Tag]]
    ThrottlingBurstLimit: Optional[int]
    ThrottlingRateLimit: Optional[float]
    TracingEnabled: Optional[bool]
    Variables: Optional[dict]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayDeploymentProvider(ResourceProvider[ApiGatewayDeploymentProperties]):
    TYPE = 'AWS::ApiGateway::Deployment'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayDeploymentProperties]) -> ProgressEvent[ApiGatewayDeploymentProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/DeploymentId\n          - /properties/RestApiId\n\n        Required properties:\n          - RestApiId\n\n        Create-only properties:\n          - /properties/DeploymentCanarySettings\n          - /properties/RestApiId\n\n        Read-only properties:\n          - /properties/DeploymentId\n\n        IAM permissions required:\n          - apigateway:POST\n\n        '
        model = request.desired_state
        api = request.aws_client_factory.apigateway
        params = {'restApiId': model['RestApiId']}
        if model.get('StageName'):
            params['stageName'] = model['StageName']
        if model.get('StageDescription'):
            params['stageDescription'] = model['StageDescription']
        if model.get('Description'):
            params['description'] = model['Description']
        response = api.create_deployment(**params)
        model['DeploymentId'] = response['id']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayDeploymentProperties]) -> ProgressEvent[ApiGatewayDeploymentProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayDeploymentProperties]) -> ProgressEvent[ApiGatewayDeploymentProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:GET\n          - apigateway:DELETE\n        '
        model = request.desired_state
        api = request.aws_client_factory.apigateway
        try:
            api.delete_deployment(restApiId=model['RestApiId'], deploymentId=model['DeploymentId'])
        except api.exceptions.NotFoundException:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayDeploymentProperties]) -> ProgressEvent[ApiGatewayDeploymentProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:PATCH\n          - apigateway:GET\n          - apigateway:PUT\n        '
        raise NotImplementedError