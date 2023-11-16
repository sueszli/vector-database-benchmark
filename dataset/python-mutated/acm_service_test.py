import uuid
from datetime import datetime
import botocore
from boto3 import session
from freezegun import freeze_time
from mock import patch
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.acm.acm_service import ACM
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'us-east-1'
make_api_call = botocore.client.BaseClient._make_api_call
certificate_arn = f'arn:aws:acm:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:certificate/{str(uuid.uuid4())}'
certificate_name = 'test-certificate.com'
certificate_type = 'AMAZON_ISSUED'

def mock_make_api_call(self, operation_name, kwargs):
    if False:
        while True:
            i = 10
    '\n    As you can see the operation_name has the list_analyzers snake_case form but\n    we are using the ListAnalyzers form.\n    Rationale -> https://github.com/boto/botocore/blob/develop/botocore/client.py#L810:L816\n\n    We have to mock every AWS API call using Boto3\n    '
    if operation_name == 'ListCertificates':
        return {'CertificateSummaryList': [{'CertificateArn': certificate_arn, 'DomainName': certificate_name, 'SubjectAlternativeNameSummaries': ['test-certificate-2.com'], 'HasAdditionalSubjectAlternativeNames': False, 'Status': 'ISSUED', 'Type': certificate_type, 'KeyAlgorithm': 'RSA_4096', 'KeyUsages': ['DIGITAL_SIGNATURE'], 'ExtendedKeyUsages': ['TLS_WEB_SERVER_AUTHENTICATION'], 'InUse': True, 'Exported': False, 'RenewalEligibility': 'ELIGIBLE', 'NotBefore': datetime(2024, 1, 1), 'NotAfter': datetime(2024, 1, 1), 'CreatedAt': datetime(2024, 1, 1), 'IssuedAt': datetime(2024, 1, 1), 'ImportedAt': datetime(2024, 1, 1), 'RevokedAt': datetime(2024, 1, 1)}]}
    if operation_name == 'DescribeCertificate':
        if kwargs['CertificateArn'] == certificate_arn:
            return {'Certificate': {'Options': {'CertificateTransparencyLoggingPreference': 'DISABLED'}}}
    if operation_name == 'ListTagsForCertificate':
        if kwargs['CertificateArn'] == certificate_arn:
            return {'Tags': [{'Key': 'test', 'Value': 'test'}]}
    return make_api_call(self, operation_name, kwargs)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        return 10
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
@freeze_time('2023-01-01')
class Test_ACM_Service:

    def set_mocked_audit_info(self):
        if False:
            print('Hello World!')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test_service(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        acm = ACM(audit_info)
        assert acm.service == 'acm'

    def test_client(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        acm = ACM(audit_info)
        for regional_client in acm.regional_clients.values():
            assert regional_client.__class__.__name__ == 'ACM'

    def test__get_session__(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        acm = ACM(audit_info)
        assert acm.session.__class__.__name__ == 'Session'

    def test_audited_account(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        acm = ACM(audit_info)
        assert acm.audited_account == AWS_ACCOUNT_NUMBER

    def test__list_and_describe_certificates__(self):
        if False:
            return 10
        audit_info = self.set_mocked_audit_info()
        acm = ACM(audit_info)
        assert len(acm.certificates) == 1
        assert acm.certificates[0].arn == certificate_arn
        assert acm.certificates[0].name == certificate_name
        assert acm.certificates[0].type == certificate_type
        assert acm.certificates[0].expiration_days == 365
        assert acm.certificates[0].transparency_logging is False
        assert acm.certificates[0].region == AWS_REGION

    def test__list_tags_for_certificate__(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        acm = ACM(audit_info)
        assert len(acm.certificates) == 1
        assert acm.certificates[0].tags == [{'Key': 'test', 'Value': 'test'}]