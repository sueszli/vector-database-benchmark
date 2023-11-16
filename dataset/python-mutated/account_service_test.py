import botocore
from boto3 import session
from mock import patch
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.account.account_service import Account, Contact
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_ACCOUNT_ARN = f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'
AWS_REGION = 'us-east-1'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwargs):
    if False:
        i = 10
        return i + 15
    '\n    As you can see the operation_name has the list_analyzers snake_case form but\n    we are using the ListAnalyzers form.\n    Rationale -> https://github.com/boto/botocore/blob/develop/botocore/client.py#L810:L816\n\n    We have to mock every AWS API call using Boto3\n    '
    if operation_name == 'GetContactInformation':
        return {'ContactInformation': {'AddressLine1': 'AddressLine1', 'AddressLine2': 'AddressLine2', 'AddressLine3': 'AddressLine3', 'City': 'City', 'CompanyName': 'Prowler', 'CountryCode': 'CountryCode', 'DistrictOrCounty': 'DistrictOrCounty', 'FullName': 'Prowler', 'PhoneNumber': '666666666', 'PostalCode': 'PostalCode', 'StateOrRegion': 'StateOrRegion', 'WebsiteUrl': 'WebsiteUrl'}}
    if operation_name == 'GetAlternateContact':
        return {'AlternateContact': {'AlternateContactType': 'SECURITY', 'EmailAddress': 'test@test.com', 'Name': 'Prowler', 'PhoneNumber': '666666666', 'Title': 'Title'}}
    return make_api_call(self, operation_name, kwargs)

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
class Test_Account_Service:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=AWS_ACCOUNT_ARN, audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test_service(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        account = Account(audit_info)
        assert account.service == 'account'

    def test_client(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        account = Account(audit_info)
        assert account.client.__class__.__name__ == 'Account'

    def test__get_session__(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        account = Account(audit_info)
        assert account.session.__class__.__name__ == 'Session'

    def test_audited_account(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        account = Account(audit_info)
        assert account.audited_account == AWS_ACCOUNT_NUMBER

    def test_get_account_contacts(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        account = Account(audit_info)
        assert account.number_of_contacts == 4
        assert account.contact_base == Contact(type='PRIMARY', name='Prowler', phone_number='666666666')
        assert account.contacts_billing == Contact(type='BILLING', email='test@test.com', name='Prowler', phone_number='666666666')
        assert account.contacts_security == Contact(type='SECURITY', email='test@test.com', name='Prowler', phone_number='666666666')
        assert account.contacts_operations == Contact(type='OPERATIONS', email='test@test.com', name='Prowler', phone_number='666666666')