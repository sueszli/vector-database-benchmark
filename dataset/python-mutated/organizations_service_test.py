import json
from boto3 import client, session
from moto import mock_organizations
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.lib.audit_info.audit_info import AWS_Audit_Info
from prowler.providers.aws.services.organizations.organizations_service import Organizations
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'

def scp_restrict_regions_with_deny():
    if False:
        for i in range(10):
            print('nop')
    return '{"Version":"2012-10-17","Statement":{"Effect":"Deny","NotAction":"s3:*","Resource":"*","Condition":{"StringNotEquals":{"aws:RequestedRegion":["eu-central-1"]}}}}'

class Test_Organizations_Service:

    def set_mocked_audit_info(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None, region_name=AWS_REGION), audited_account=DEFAULT_ACCOUNT_ID, audited_account_arn=f'arn:aws:iam::{DEFAULT_ACCOUNT_ID}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=AWS_REGION, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_organizations
    def test_service(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        organizations = Organizations(audit_info)
        assert organizations.service == 'organizations'

    @mock_organizations
    def test__describe_organization__(self):
        if False:
            print('Hello World!')
        conn = client('organizations', region_name=AWS_REGION)
        response = conn.create_organization()
        audit_info = self.set_mocked_audit_info()
        organizations = Organizations(audit_info)
        assert len(organizations.organizations) == 1
        assert organizations.organizations[0].arn == response['Organization']['Arn']
        assert organizations.organizations[0].id == response['Organization']['Id']
        assert organizations.organizations[0].master_id == response['Organization']['MasterAccountId']
        assert organizations.organizations[0].status == 'ACTIVE'
        assert organizations.organizations[0].delegated_administrators == []

    @mock_organizations
    def test__list_policies__(self):
        if False:
            for i in range(10):
                print('nop')
        conn = client('organizations', region_name=AWS_REGION)
        conn.create_organization()
        response = conn.create_policy(Content=scp_restrict_regions_with_deny(), Description='Test', Name='Test', Type='SERVICE_CONTROL_POLICY')
        audit_info = self.set_mocked_audit_info()
        organizations = Organizations(audit_info)
        for policy in organizations.policies:
            if policy.arn == response['Policy']['PolicySummary']['Arn']:
                assert policy.type == 'SERVICE_CONTROL_POLICY'
                assert policy.aws_managed is False
                assert policy.content == json.loads(response['Policy']['Content'])
                assert policy.targets == []