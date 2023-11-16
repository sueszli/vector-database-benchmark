import datetime
from unittest.mock import patch
import botocore
from boto3 import session
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.macie.macie_service import Macie, Session
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'
AWS_ACCOUNT_NUMBER = '123456789012'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        while True:
            i = 10
    if operation_name == 'GetMacieSession':
        return {'createdAt': datetime(2015, 1, 1), 'findingPublishingFrequency': 'SIX_HOURS', 'serviceRole': 'string', 'status': 'ENABLED', 'updatedAt': datetime(2015, 1, 1)}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        while True:
            i = 10
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_Macie_Service:

    def set_mocked_audit_info(self):
        if False:
            print('Hello World!')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test__get_client__(self):
        if False:
            for i in range(10):
                print('nop')
        macie = Macie(self.set_mocked_audit_info())
        assert macie.regional_clients[AWS_REGION].__class__.__name__ == 'Macie2'

    def test__get_session__(self):
        if False:
            i = 10
            return i + 15
        macie = Macie(self.set_mocked_audit_info())
        assert macie.session.__class__.__name__ == 'Session'

    def test__get_service__(self):
        if False:
            i = 10
            return i + 15
        macie = Macie(self.set_mocked_audit_info())
        assert macie.service == 'macie2'

    def test__get_macie_session__(self):
        if False:
            i = 10
            return i + 15
        macie = Macie(self.set_mocked_audit_info())
        macie.sessions = [Session(status='ENABLED', region='eu-west-1')]
        assert len(macie.sessions) == 1
        assert macie.sessions[0].status == 'ENABLED'
        assert macie.sessions[0].region == AWS_REGION