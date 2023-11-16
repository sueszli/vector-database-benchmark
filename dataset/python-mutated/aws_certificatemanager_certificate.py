from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class CertificateManagerCertificateProperties(TypedDict):
    DomainName: Optional[str]
    CertificateAuthorityArn: Optional[str]
    CertificateTransparencyLoggingPreference: Optional[str]
    DomainValidationOptions: Optional[list[DomainValidationOption]]
    Id: Optional[str]
    SubjectAlternativeNames: Optional[list[str]]
    Tags: Optional[list[Tag]]
    ValidationMethod: Optional[str]

class DomainValidationOption(TypedDict):
    DomainName: Optional[str]
    HostedZoneId: Optional[str]
    ValidationDomain: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class CertificateManagerCertificateProvider(ResourceProvider[CertificateManagerCertificateProperties]):
    TYPE = 'AWS::CertificateManager::Certificate'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[CertificateManagerCertificateProperties]) -> ProgressEvent[CertificateManagerCertificateProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - DomainName\n\n        Create-only properties:\n          - /properties/SubjectAlternativeNames\n          - /properties/DomainValidationOptions\n          - /properties/ValidationMethod\n          - /properties/DomainName\n          - /properties/CertificateAuthorityArn\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        acm = request.aws_client_factory.acm
        params = util.select_attributes(model, ['CertificateAuthorityArn', 'DomainName', 'DomainValidationOptions', 'SubjectAlternativeNames', 'Tags', 'ValidationMethod'])
        valid_opts = params.get('DomainValidationOptions')
        if valid_opts:

            def _convert(opt):
                if False:
                    print('Hello World!')
                res = util.select_attributes(opt, ['DomainName', 'ValidationDomain'])
                res.setdefault('ValidationDomain', res['DomainName'])
                return res
            params['DomainValidationOptions'] = [_convert(opt) for opt in valid_opts]
        logging_pref = params.get('CertificateTransparencyLoggingPreference')
        if logging_pref:
            params['Options'] = {'CertificateTransparencyLoggingPreference': logging_pref}
        response = acm.request_certificate(**params)
        model['Id'] = response['CertificateArn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[CertificateManagerCertificateProperties]) -> ProgressEvent[CertificateManagerCertificateProperties]:
        if False:
            return 10
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[CertificateManagerCertificateProperties]) -> ProgressEvent[CertificateManagerCertificateProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        acm = request.aws_client_factory.acm
        acm.delete_certificate(CertificateArn=model['Id'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[CertificateManagerCertificateProperties]) -> ProgressEvent[CertificateManagerCertificateProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError