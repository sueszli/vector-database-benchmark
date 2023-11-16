from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.objects import keys_to_lower

class ApiGatewayDomainNameProperties(TypedDict):
    CertificateArn: Optional[str]
    DistributionDomainName: Optional[str]
    DistributionHostedZoneId: Optional[str]
    DomainName: Optional[str]
    EndpointConfiguration: Optional[EndpointConfiguration]
    MutualTlsAuthentication: Optional[MutualTlsAuthentication]
    OwnershipVerificationCertificateArn: Optional[str]
    RegionalCertificateArn: Optional[str]
    RegionalDomainName: Optional[str]
    RegionalHostedZoneId: Optional[str]
    SecurityPolicy: Optional[str]
    Tags: Optional[list[Tag]]

class EndpointConfiguration(TypedDict):
    Types: Optional[list[str]]

class MutualTlsAuthentication(TypedDict):
    TruststoreUri: Optional[str]
    TruststoreVersion: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class ApiGatewayDomainNameProvider(ResourceProvider[ApiGatewayDomainNameProperties]):
    TYPE = 'AWS::ApiGateway::DomainName'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ApiGatewayDomainNameProperties]) -> ProgressEvent[ApiGatewayDomainNameProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/DomainName\n\n        Create-only properties:\n          - /properties/DomainName\n\n        Read-only properties:\n          - /properties/RegionalHostedZoneId\n          - /properties/DistributionDomainName\n          - /properties/RegionalDomainName\n          - /properties/DistributionHostedZoneId\n\n        IAM permissions required:\n          - apigateway:*\n\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        params = keys_to_lower(model.copy())
        param_names = ['certificateArn', 'domainName', 'endpointConfiguration', 'mutualTlsAuthentication', 'ownershipVerificationCertificateArn', 'regionalCertificateArn', 'securityPolicy']
        params = util.select_attributes(params, param_names)
        if model.get('Tags'):
            params['tags'] = {tag['key']: tag['value'] for tag in model['Tags']}
        result = apigw.create_domain_name(**params)
        hosted_zones = request.aws_client_factory.route53.list_hosted_zones()
        '\n        The hardcoded value is the only one that should be returned but due limitations it is not possible to\n        use it.\n        '
        if hosted_zones['HostedZones']:
            model['DistributionHostedZoneId'] = hosted_zones['HostedZones'][0]['Id']
        else:
            model['DistributionHostedZoneId'] = 'Z2FDTNDATAQYW2'
        model['DistributionDomainName'] = result.get('distributionDomainName') or result.get('domainName')
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ApiGatewayDomainNameProperties]) -> ProgressEvent[ApiGatewayDomainNameProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - apigateway:*\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ApiGatewayDomainNameProperties]) -> ProgressEvent[ApiGatewayDomainNameProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - apigateway:*\n        '
        model = request.desired_state
        apigw = request.aws_client_factory.apigateway
        apigw.delete_domain_name(domainName=model['DomainName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ApiGatewayDomainNameProperties]) -> ProgressEvent[ApiGatewayDomainNameProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - apigateway:*\n        '
        raise NotImplementedError