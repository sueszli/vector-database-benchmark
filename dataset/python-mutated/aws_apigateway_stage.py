from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.objects import keys_to_lower

class ApiGatewayStageProperties(TypedDict):
    RestApiId: Optional[str]
    AccessLogSetting: Optional[AccessLogSetting]
    CacheClusterEnabled: Optional[bool]
    CacheClusterSize: Optional[str]
    CanarySetting: Optional[CanarySetting]
    ClientCertificateId: Optional[str]
    DeploymentId: Optional[str]
    Description: Optional[str]
    DocumentationVersion: Optional[str]
    MethodSettings: Optional[list[MethodSetting]]
    StageName: Optional[str]
    Tags: Optional[list[Tag]]
    TracingEnabled: Optional[bool]
    Variables: Optional[dict]

class AccessLogSetting(TypedDict):
    DestinationArn: Optional[str]
    Format: Optional[str]

class CanarySetting(TypedDict):
    DeploymentId: Optional[str]
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
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayStageProvider(ResourceProvider[ApiGatewayStageProperties]):
    TYPE = 'AWS::ApiGateway::Stage'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayStageProperties]) -> ProgressEvent[ApiGatewayStageProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/RestApiId\n          - /properties/StageName\n\n        Required properties:\n          - RestApiId\n\n        Create-only properties:\n          - /properties/RestApiId\n          - /properties/StageName\n\n\n\n        IAM permissions required:\n          - apigateway:POST\n          - apigateway:PATCH\n          - apigateway:GET\n\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        stage_name = model.get('StageName', 'default')
        params = keys_to_lower(model.copy())
        param_names = ['restApiId', 'deploymentId', 'description', 'cacheClusterEnabled', 'cacheClusterSize', 'variables', 'documentationVersion', 'canarySettings', 'tracingEnabled', 'tags']
        params = util.select_attributes(params, param_names)
        params['tags'] = {t['key']: t['value'] for t in params.get('tags', [])}
        params['stageName'] = stage_name
        result = apigw.create_stage(**params)
        model['StageName'] = result['stageName']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayStageProperties]) -> ProgressEvent[ApiGatewayStageProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:GET\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayStageProperties]) -> ProgressEvent[ApiGatewayStageProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:DELETE\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        try:
            apigw.get_stage(restApiId=model['RestApiId'], stageName=model['StageName'])
            apigw.delete_stage(restApiId=model['RestApiId'], stageName=model['StageName'])
        except apigw.exceptions.NotFoundException:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayStageProperties]) -> ProgressEvent[ApiGatewayStageProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:GET\n          - apigateway:PATCH\n          - apigateway:PUT\n          - apigateway:DELETE\n        '
        raise NotImplementedError