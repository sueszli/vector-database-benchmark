from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class LambdaAliasProperties(TypedDict):
    FunctionName: Optional[str]
    FunctionVersion: Optional[str]
    Name: Optional[str]
    Description: Optional[str]
    Id: Optional[str]
    ProvisionedConcurrencyConfig: Optional[ProvisionedConcurrencyConfiguration]
    RoutingConfig: Optional[AliasRoutingConfiguration]

class ProvisionedConcurrencyConfiguration(TypedDict):
    ProvisionedConcurrentExecutions: Optional[int]

class VersionWeight(TypedDict):
    FunctionVersion: Optional[str]
    FunctionWeight: Optional[float]

class AliasRoutingConfiguration(TypedDict):
    AdditionalVersionWeights: Optional[list[VersionWeight]]
REPEATED_INVOCATION = 'repeated_invocation'

class LambdaAliasProvider(ResourceProvider[LambdaAliasProperties]):
    TYPE = 'AWS::Lambda::Alias'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[LambdaAliasProperties]) -> ProgressEvent[LambdaAliasProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - FunctionName\n          - FunctionVersion\n          - Name\n\n        Create-only properties:\n          - /properties/Name\n          - /properties/FunctionName\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        lambda_ = request.aws_client_factory.lambda_
        create_params = util.select_attributes(model, ['FunctionName', 'FunctionVersion', 'Name', 'Description', 'RoutingConfig'])
        ctx = request.custom_context
        if not ctx.get(REPEATED_INVOCATION):
            result = lambda_.create_alias(**create_params)
            model['Id'] = result['AliasArn']
            ctx[REPEATED_INVOCATION] = True
            if model.get('ProvisionedConcurrencyConfig'):
                lambda_.put_provisioned_concurrency_config(FunctionName=model['FunctionName'], Qualifier=model['Id'].split(':')[-1], ProvisionedConcurrentExecutions=model['ProvisionedConcurrencyConfig']['ProvisionedConcurrentExecutions'])
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=ctx)
        if ctx.get(REPEATED_INVOCATION) and model.get('ProvisionedConcurrencyConfig'):
            result = lambda_.get_provisioned_concurrency_config(FunctionName=model['FunctionName'], Qualifier=model['Id'].split(':')[-1])
            if result['Status'] == 'IN_PROGRESS':
                return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[LambdaAliasProperties]) -> ProgressEvent[LambdaAliasProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[LambdaAliasProperties]) -> ProgressEvent[LambdaAliasProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        lambda_ = request.aws_client_factory.lambda_
        try:
            lambda_.delete_alias(FunctionName=model['FunctionName'], Name=model['Name'])
        except lambda_.exceptions.ResourceNotFoundException:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=None)

    def update(self, request: ResourceRequest[LambdaAliasProperties]) -> ProgressEvent[LambdaAliasProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError