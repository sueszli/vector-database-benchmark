from re import search
import boto3
from mock import patch
from moto import mock_iam, mock_sts
from prowler.providers.aws.aws_provider import AWS_Provider, assume_role, generate_regional_clients, get_available_aws_service_regions, get_default_region, get_global_region
from prowler.providers.aws.lib.audit_info.models import AWS_Assume_Role, AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
ACCOUNT_ID = 123456789012
AWS_REGION = 'us-east-1'

class Test_AWS_Provider:

    @mock_iam
    @mock_sts
    def test_aws_provider_user_without_mfa(self):
        if False:
            return 10
        audited_regions = ['eu-west-1']
        iam_client = boto3.client('iam', region_name=AWS_REGION)
        iam_user = iam_client.create_user(UserName='test-user')['User']
        access_key = iam_client.create_access_key(UserName=iam_user['UserName'])['AccessKey']
        access_key_id = access_key['AccessKeyId']
        secret_access_key = access_key['SecretAccessKey']
        session = boto3.session.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name=AWS_REGION)
        audit_info = AWS_Audit_Info(session_config=None, original_session=session, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition=None, audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=AWS_Assume_Role(role_arn=None, session_duration=None, external_id=None, mfa_enabled=False), audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        with patch('prowler.providers.aws.aws_provider.input_role_mfa_token_and_code', return_value=(f'arn:aws:iam::{ACCOUNT_ID}:mfa/test-role-mfa', '111111')):
            aws_provider = AWS_Provider(audit_info)
            assert aws_provider.aws_session.region_name is None
            assert aws_provider.role_info == AWS_Assume_Role(role_arn=None, session_duration=None, external_id=None, mfa_enabled=False)

    @mock_iam
    @mock_sts
    def test_aws_provider_user_with_mfa(self):
        if False:
            i = 10
            return i + 15
        audited_regions = 'eu-west-1'
        iam_client = boto3.client('iam', region_name=AWS_REGION)
        iam_user = iam_client.create_user(UserName='test-user')['User']
        access_key = iam_client.create_access_key(UserName=iam_user['UserName'])['AccessKey']
        access_key_id = access_key['AccessKeyId']
        secret_access_key = access_key['SecretAccessKey']
        session = boto3.session.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name=AWS_REGION)
        audit_info = AWS_Audit_Info(session_config=None, original_session=session, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition=None, audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=AWS_REGION, credentials=None, assumed_role_info=AWS_Assume_Role(role_arn=None, session_duration=None, external_id=None, mfa_enabled=False), audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=True)
        with patch('prowler.providers.aws.aws_provider.input_role_mfa_token_and_code', return_value=(f'arn:aws:iam::{ACCOUNT_ID}:mfa/test-role-mfa', '111111')):
            aws_provider = AWS_Provider(audit_info)
            assert aws_provider.aws_session.region_name is None
            assert aws_provider.role_info == AWS_Assume_Role(role_arn=None, session_duration=None, external_id=None, mfa_enabled=False)

    @mock_iam
    @mock_sts
    def test_aws_provider_assume_role_with_mfa(self):
        if False:
            while True:
                i = 10
        role_name = 'test-role'
        role_arn = f'arn:aws:iam::{ACCOUNT_ID}:role/{role_name}'
        session_duration_seconds = 900
        audited_regions = ['eu-west-1']
        sessionName = 'ProwlerAsessmentSession'
        iam_client = boto3.client('iam', region_name=AWS_REGION)
        iam_user = iam_client.create_user(UserName='test-user')['User']
        access_key = iam_client.create_access_key(UserName=iam_user['UserName'])['AccessKey']
        access_key_id = access_key['AccessKeyId']
        secret_access_key = access_key['SecretAccessKey']
        session = boto3.session.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name=AWS_REGION)
        audit_info = AWS_Audit_Info(session_config=None, original_session=session, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition=None, audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=AWS_Assume_Role(role_arn=role_arn, session_duration=session_duration_seconds, external_id=None, mfa_enabled=True), audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        aws_provider = AWS_Provider(audit_info)
        with patch('prowler.providers.aws.aws_provider.input_role_mfa_token_and_code', return_value=(f'arn:aws:iam::{ACCOUNT_ID}:mfa/test-role-mfa', '111111')):
            assume_role_response = assume_role(aws_provider.aws_session, aws_provider.role_info)
            credentials = assume_role_response['Credentials']
            assert len(credentials['SessionToken']) == 356
            assert search('^FQoGZXIvYXdzE.*$', credentials['SessionToken'])
            assert len(credentials['AccessKeyId']) == 20
            assert search('^ASIA.*$', credentials['AccessKeyId'])
            assert len(credentials['SecretAccessKey']) == 40
            assert assume_role_response['AssumedRoleUser']['Arn'] == f'arn:aws:sts::{ACCOUNT_ID}:assumed-role/{role_name}/{sessionName}'
            assert search('^AROA.*$', assume_role_response['AssumedRoleUser']['AssumedRoleId'])
            assert search(f'^.*:{sessionName}$', assume_role_response['AssumedRoleUser']['AssumedRoleId'])
            assert len(assume_role_response['AssumedRoleUser']['AssumedRoleId']) == 21 + 1 + len(sessionName)

    @mock_iam
    @mock_sts
    def test_aws_provider_assume_role_without_mfa(self):
        if False:
            i = 10
            return i + 15
        role_name = 'test-role'
        role_arn = f'arn:aws:iam::{ACCOUNT_ID}:role/{role_name}'
        session_duration_seconds = 900
        audited_regions = 'eu-west-1'
        sessionName = 'ProwlerAsessmentSession'
        iam_client = boto3.client('iam', region_name=AWS_REGION)
        iam_user = iam_client.create_user(UserName='test-user')['User']
        access_key = iam_client.create_access_key(UserName=iam_user['UserName'])['AccessKey']
        access_key_id = access_key['AccessKeyId']
        secret_access_key = access_key['SecretAccessKey']
        session = boto3.session.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name=AWS_REGION)
        audit_info = AWS_Audit_Info(session_config=None, original_session=session, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition=None, audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=AWS_Assume_Role(role_arn=role_arn, session_duration=session_duration_seconds, external_id=None, mfa_enabled=False), audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        aws_provider = AWS_Provider(audit_info)
        assume_role_response = assume_role(aws_provider.aws_session, aws_provider.role_info)
        credentials = assume_role_response['Credentials']
        assert len(credentials['SessionToken']) == 356
        assert search('^FQoGZXIvYXdzE.*$', credentials['SessionToken'])
        assert len(credentials['AccessKeyId']) == 20
        assert search('^ASIA.*$', credentials['AccessKeyId'])
        assert len(credentials['SecretAccessKey']) == 40
        assert assume_role_response['AssumedRoleUser']['Arn'] == f'arn:aws:sts::{ACCOUNT_ID}:assumed-role/{role_name}/{sessionName}'
        assert search('^AROA.*$', assume_role_response['AssumedRoleUser']['AssumedRoleId'])
        assert search(f'^.*:{sessionName}$', assume_role_response['AssumedRoleUser']['AssumedRoleId'])
        assert len(assume_role_response['AssumedRoleUser']['AssumedRoleId']) == 21 + 1 + len(sessionName)

    @mock_iam
    @mock_sts
    def test_assume_role_with_sts_endpoint_region(self):
        if False:
            i = 10
            return i + 15
        role_name = 'test-role'
        role_arn = f'arn:aws:iam::{ACCOUNT_ID}:role/{role_name}'
        session_duration_seconds = 900
        aws_region = 'eu-west-1'
        sts_endpoint_region = aws_region
        audited_regions = [aws_region]
        sessionName = 'ProwlerAsessmentSession'
        iam_client = boto3.client('iam', region_name=AWS_REGION)
        iam_user = iam_client.create_user(UserName='test-user')['User']
        access_key = iam_client.create_access_key(UserName=iam_user['UserName'])['AccessKey']
        access_key_id = access_key['AccessKeyId']
        secret_access_key = access_key['SecretAccessKey']
        session = boto3.session.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name=AWS_REGION)
        audit_info = AWS_Audit_Info(session_config=None, original_session=session, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition=None, audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=AWS_Assume_Role(role_arn=role_arn, session_duration=session_duration_seconds, external_id=None, mfa_enabled=False), audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        aws_provider = AWS_Provider(audit_info)
        assume_role_response = assume_role(aws_provider.aws_session, aws_provider.role_info, sts_endpoint_region)
        credentials = assume_role_response['Credentials']
        assert len(credentials['SessionToken']) == 356
        assert search('^FQoGZXIvYXdzE.*$', credentials['SessionToken'])
        assert len(credentials['AccessKeyId']) == 20
        assert search('^ASIA.*$', credentials['AccessKeyId'])
        assert len(credentials['SecretAccessKey']) == 40
        assert assume_role_response['AssumedRoleUser']['Arn'] == f'arn:aws:sts::{ACCOUNT_ID}:assumed-role/{role_name}/{sessionName}'
        assert search('^AROA.*$', assume_role_response['AssumedRoleUser']['AssumedRoleId'])
        assert search(f'^.*:{sessionName}$', assume_role_response['AssumedRoleUser']['AssumedRoleId'])
        assert len(assume_role_response['AssumedRoleUser']['AssumedRoleId']) == 21 + 1 + len(sessionName)

    def test_generate_regional_clients(self):
        if False:
            i = 10
            return i + 15
        session = boto3.session.Session(region_name=AWS_REGION)
        audited_regions = ['eu-west-1', AWS_REGION]
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        generate_regional_clients_response = generate_regional_clients('ec2', audit_info)
        assert set(generate_regional_clients_response.keys()) == set(audited_regions)

    def test_generate_regional_clients_global_service(self):
        if False:
            for i in range(10):
                print('nop')
        session = boto3.session.Session(region_name=AWS_REGION)
        audited_regions = ['eu-west-1', AWS_REGION]
        profile_region = AWS_REGION
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=profile_region, credentials=None, assumed_role_info=None, audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        generate_regional_clients_response = generate_regional_clients('route53', audit_info, global_service=True)
        assert list(generate_regional_clients_response.keys()) == [profile_region]

    def test_generate_regional_clients_cn_partition(self):
        if False:
            for i in range(10):
                print('nop')
        session = boto3.session.Session(region_name=AWS_REGION)
        audited_regions = ['cn-northwest-1', 'cn-north-1']
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session, audited_account=None, audited_account_arn=None, audited_partition='aws-cn', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        generate_regional_clients_response = generate_regional_clients('shield', audit_info, global_service=True)
        assert generate_regional_clients_response == {}

    def test_get_default_region(self):
        if False:
            for i in range(10):
                print('nop')
        audited_regions = ['eu-west-1']
        profile_region = 'eu-west-1'
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=profile_region, credentials=None, assumed_role_info=None, audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        assert get_default_region('ec2', audit_info) == 'eu-west-1'

    def test_get_default_region_profile_region_not_audited(self):
        if False:
            i = 10
            return i + 15
        audited_regions = ['eu-west-1']
        profile_region = 'us-east-2'
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=profile_region, credentials=None, assumed_role_info=None, audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        assert get_default_region('ec2', audit_info) == 'eu-west-1'

    def test_get_default_region_non_profile_region(self):
        if False:
            i = 10
            return i + 15
        audited_regions = ['eu-west-1']
        profile_region = None
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=profile_region, credentials=None, assumed_role_info=None, audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        assert get_default_region('ec2', audit_info) == 'eu-west-1'

    def test_get_default_region_non_profile_or_audited_region(self):
        if False:
            print('Hello World!')
        audited_regions = None
        profile_region = None
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=profile_region, credentials=None, assumed_role_info=None, audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        assert get_default_region('ec2', audit_info) == 'us-east-1'

    def test_aws_get_global_region(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        assert get_default_region('ec2', audit_info) == 'us-east-1'

    def test_aws_gov_get_global_region(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws-us-gov', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        assert get_global_region(audit_info) == 'us-gov-east-1'

    def test_aws_cn_get_global_region(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws-cn', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        assert get_global_region(audit_info) == 'cn-north-1'

    def test_aws_iso_get_global_region(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws-iso', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        assert get_global_region(audit_info) == 'aws-iso-global'

    def test_get_available_aws_service_regions_with_us_east_1_audited(self):
        if False:
            i = 10
            return i + 15
        audited_regions = ['us-east-1']
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=audited_regions, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        with patch('prowler.providers.aws.aws_provider.parse_json_file', return_value={'services': {'ec2': {'regions': {'aws': ['af-south-1', 'ca-central-1', 'eu-central-1', 'eu-central-2', 'eu-north-1', 'eu-south-1', 'eu-south-2', 'eu-west-1', 'eu-west-2', 'eu-west-3', 'me-central-1', 'me-south-1', 'sa-east-1', 'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2']}}}}):
            assert get_available_aws_service_regions('ec2', audit_info) == ['us-east-1']

    def test_get_available_aws_service_regions_with_all_regions_audited(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=None, audited_account=None, audited_account_arn=None, audited_partition='aws', audited_identity_arn=None, audited_user_id=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        with patch('prowler.providers.aws.aws_provider.parse_json_file', return_value={'services': {'ec2': {'regions': {'aws': ['af-south-1', 'ca-central-1', 'eu-central-1', 'eu-central-2', 'eu-north-1', 'eu-south-1', 'eu-south-2', 'eu-west-1', 'eu-west-2', 'eu-west-3', 'me-central-1', 'me-south-1', 'sa-east-1', 'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2']}}}}):
            assert len(get_available_aws_service_regions('ec2', audit_info)) == 17