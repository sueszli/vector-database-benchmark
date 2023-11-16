from json import dumps
from uuid import uuid4
import botocore
from boto3 import client, session
from freezegun import freeze_time
from mock import patch
from moto import mock_iam
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.iam.iam_service import IAM, Policy, is_service_role
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
TEST_DATETIME = '2023-01-01T12:01:01+00:00'
INLINE_POLICY_NOT_ADMIN = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['s3:GetObject'], 'Resource': '*'}]}
ASSUME_ROLE_POLICY_DOCUMENT = {'Version': '2012-10-17', 'Statement': {'Sid': 'test', 'Effect': 'Allow', 'Principal': {'AWS': f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'}, 'Action': 'sts:AssumeRole'}}
SECURITY_AUDIT_POLICY_ARN = 'arn:aws:iam::aws:policy/SecurityAudit'
READ_ONLY_ACCESS_POLICY_ARN = 'arn:aws:iam::aws:policy/ReadOnlyAccess'
SUPPORT_SERVICE_ROLE_POLICY_ARN = 'arn:aws:iam::aws:policy/aws-service-role/AWSSupportServiceRolePolicy'
ADMINISTRATOR_ACCESS_POLICY_ARN = 'arn:aws:iam::aws:policy/AdministratorAccess'
make_api_call = botocore.client.BaseClient._make_api_call
IAM_LAST_ACCESSED_SERVICES = [{'ServiceName': 'AWS EC2', 'ServiceNamespace': 'ec2', 'TotalAuthenticatedEntities': 1}, {'ServiceName': 'AWS Identity and Access Management', 'ServiceNamespace': 'iam', 'TotalAuthenticatedEntities': 0}]

def mock_make_api_call(self, operation_name, kwargs):
    if False:
        print('Hello World!')
    '\n    As you can see the operation_name has the list_analyzers snake_case form but\n    we are using the ListAnalyzers form.\n    Rationale -> https://github.com/boto/botocore/blob/develop/botocore/client.py#L810:L816\n    We have to mock every AWS API call using Boto3\n    '
    if operation_name == 'GenerateServiceLastAccessedDetails':
        return {'JobId': str(uuid4())}
    if operation_name == 'GetServiceLastAccessedDetails':
        return {'JobStatus': 'COMPLETED', 'JobType': 'SERVICE_LEVEL', 'JobCreationDate': '2023-10-19T06:11:11.449000+00:00', 'ServicesLastAccessed': IAM_LAST_ACCESSED_SERVICES}
    return make_api_call(self, operation_name, kwargs)

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
class Test_IAM_Service:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=None, audited_account_arn=None, audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region='us-east-1', credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_iam
    def test__get_client__(self):
        if False:
            return 10
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert iam.client.__class__.__name__ == 'IAM'

    @mock_iam
    def test__get_session__(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert iam.session.__class__.__name__ == 'Session'

    @freeze_time(TEST_DATETIME)
    @mock_iam
    def test__get_credential_report__(self):
        if False:
            while True:
                i = 10
        iam_client = client('iam')
        username = 'user1'
        iam_client.create_user(UserName=username)
        expected_credential_report = {'user': username, 'arn': f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:user/{username}', 'user_creation_time': TEST_DATETIME, 'password_enabled': 'false', 'password_last_used': 'not_supported', 'password_last_changed': TEST_DATETIME, 'password_next_rotation': 'not_supported', 'mfa_active': 'false', 'access_key_1_active': 'false', 'access_key_1_last_rotated': 'N/A', 'access_key_1_last_used_date': 'N/A', 'access_key_1_last_used_region': 'not_supported', 'access_key_1_last_used_service': 'not_supported', 'access_key_2_active': 'false', 'access_key_2_last_rotated': 'N/A', 'access_key_2_last_used_date': 'N/A', 'access_key_2_last_used_region': 'not_supported', 'access_key_2_last_used_service': 'not_supported', 'cert_1_active': 'false', 'cert_1_last_rotated': 'N/A', 'cert_2_active': 'false', 'cert_2_last_rotated': 'N/A'}
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.credential_report) == 1
        assert iam.credential_report[0].get('user')
        assert iam.credential_report[0]['user'] == expected_credential_report['user']
        assert iam.credential_report[0].get('arn')
        assert iam.credential_report[0]['arn'] == expected_credential_report['arn']
        assert iam.credential_report[0].get('user_creation_time')
        assert iam.credential_report[0]['user_creation_time'] == expected_credential_report['user_creation_time']
        assert iam.credential_report[0].get('password_enabled')
        assert iam.credential_report[0]['password_enabled'] == expected_credential_report['password_enabled']
        assert iam.credential_report[0].get('password_last_used')
        assert iam.credential_report[0]['password_last_used'] == expected_credential_report['password_last_used']
        assert iam.credential_report[0].get('password_last_changed')
        assert iam.credential_report[0]['password_last_changed'] == expected_credential_report['password_last_changed']
        assert iam.credential_report[0].get('password_next_rotation')
        assert iam.credential_report[0]['password_next_rotation'] == expected_credential_report['password_next_rotation']
        assert iam.credential_report[0].get('mfa_active')
        assert iam.credential_report[0]['mfa_active'] == expected_credential_report['mfa_active']
        assert iam.credential_report[0].get('access_key_1_active')
        assert iam.credential_report[0]['access_key_1_active'] == expected_credential_report['access_key_1_active']
        assert iam.credential_report[0].get('access_key_1_last_rotated')
        assert iam.credential_report[0]['access_key_1_last_rotated'] == expected_credential_report['access_key_1_last_rotated']
        assert iam.credential_report[0].get('access_key_1_last_used_date')
        assert iam.credential_report[0]['access_key_1_last_used_date'] == expected_credential_report['access_key_1_last_used_date']
        assert iam.credential_report[0].get('access_key_1_last_used_region')
        assert iam.credential_report[0]['access_key_1_last_used_region'] == expected_credential_report['access_key_1_last_used_region']
        assert iam.credential_report[0].get('access_key_1_last_used_service')
        assert iam.credential_report[0]['access_key_1_last_used_service'] == expected_credential_report['access_key_1_last_used_service']
        assert iam.credential_report[0].get('access_key_2_active')
        assert iam.credential_report[0]['access_key_2_active'] == expected_credential_report['access_key_2_active']
        assert iam.credential_report[0].get('access_key_2_last_rotated')
        assert iam.credential_report[0]['access_key_2_last_rotated'] == expected_credential_report['access_key_2_last_rotated']
        assert iam.credential_report[0].get('access_key_2_last_used_date')
        assert iam.credential_report[0]['access_key_2_last_used_date'] == expected_credential_report['access_key_2_last_used_date']
        assert iam.credential_report[0].get('access_key_2_last_used_region')
        assert iam.credential_report[0]['access_key_2_last_used_region'] == expected_credential_report['access_key_2_last_used_region']
        assert iam.credential_report[0].get('access_key_2_last_used_service')
        assert iam.credential_report[0]['access_key_2_last_used_service'] == expected_credential_report['access_key_2_last_used_service']
        assert iam.credential_report[0].get('cert_1_active')
        assert iam.credential_report[0]['cert_1_active'] == expected_credential_report['cert_1_active']
        assert iam.credential_report[0].get('cert_1_last_rotated')
        assert iam.credential_report[0]['cert_1_last_rotated'] == expected_credential_report['cert_1_last_rotated']
        assert iam.credential_report[0].get('cert_2_active')
        assert iam.credential_report[0]['cert_2_active'] == expected_credential_report['cert_2_active']
        assert iam.credential_report[0].get('cert_2_last_rotated')
        assert iam.credential_report[0]['cert_2_last_rotated'] == expected_credential_report['cert_2_last_rotated']

    @mock_iam
    def test__get_roles__(self):
        if False:
            for i in range(10):
                print('nop')
        iam_client = client('iam')
        service_policy_document = {'Version': '2008-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'ec2.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'AWS': f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'}, 'Action': 'sts:AssumeRole'}]}
        service_role = iam_client.create_role(RoleName='test-1', AssumeRolePolicyDocument=dumps(service_policy_document), Tags=[{'Key': 'test', 'Value': 'test'}])['Role']
        role = iam_client.create_role(RoleName='test-2', AssumeRolePolicyDocument=dumps(policy_document), Tags=[{'Key': 'test', 'Value': 'test'}])['Role']
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.roles) == len(iam_client.list_roles()['Roles'])
        assert iam.roles[0].tags == [{'Key': 'test', 'Value': 'test'}]
        assert iam.roles[1].tags == [{'Key': 'test', 'Value': 'test'}]
        assert is_service_role(service_role)
        assert not is_service_role(role)

    @mock_iam
    def test__get_groups__(self):
        if False:
            for i in range(10):
                print('nop')
        iam_client = client('iam')
        iam_client.create_group(GroupName='group1')
        iam_client.create_group(GroupName='group2')
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.groups) == len(iam_client.list_groups()['Groups'])

    @mock_iam
    def test__get_users__(self):
        if False:
            return 10
        iam_client = client('iam')
        iam_client.create_user(UserName='user1', Tags=[{'Key': 'test', 'Value': 'test'}])
        iam_client.create_user(UserName='user2', Tags=[{'Key': 'test', 'Value': 'test'}])
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.users) == len(iam_client.list_users()['Users'])
        assert iam.users[0].tags == [{'Key': 'test', 'Value': 'test'}]
        assert iam.users[1].tags == [{'Key': 'test', 'Value': 'test'}]

    @mock_iam
    def test__get_account_summary__(self):
        if False:
            print('Hello World!')
        iam_client = client('iam')
        account_summary = iam_client.get_account_summary()['SummaryMap']
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert iam.account_summary['SummaryMap'] == account_summary

    @mock_iam
    def test__get_password_policy__(self):
        if False:
            print('Hello World!')
        iam_client = client('iam')
        min_password_length = 123
        require_symbols = False
        require_numbers = True
        require_upper = True
        require_lower = False
        allow_users_to_change = True
        max_password_age = 123
        password_reuse_prevention = 24
        hard_expiry = True
        iam_client.update_account_password_policy(MinimumPasswordLength=min_password_length, RequireSymbols=require_symbols, RequireNumbers=require_numbers, RequireUppercaseCharacters=require_upper, RequireLowercaseCharacters=require_lower, AllowUsersToChangePassword=allow_users_to_change, MaxPasswordAge=max_password_age, PasswordReusePrevention=password_reuse_prevention, HardExpiry=hard_expiry)
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert iam.password_policy.length == min_password_length
        assert iam.password_policy.symbols == require_symbols
        assert iam.password_policy.numbers == require_numbers
        assert iam.password_policy.uppercase == require_upper
        assert iam.password_policy.lowercase == require_lower
        assert iam.password_policy.allow_change == allow_users_to_change
        assert iam.password_policy.expiration is True
        assert iam.password_policy.max_age == max_password_age
        assert iam.password_policy.reuse_prevention == password_reuse_prevention
        assert iam.password_policy.hard_expiry == hard_expiry

    @mock_iam
    def test__list_mfa_devices__(self):
        if False:
            while True:
                i = 10
        iam_client = client('iam')
        iam_client.create_user(UserName='user1')
        mfa_device_name = 'test-mfa-device'
        virtual_mfa_device = iam_client.create_virtual_mfa_device(VirtualMFADeviceName=mfa_device_name)
        iam_client.enable_mfa_device(UserName='user1', SerialNumber=virtual_mfa_device['VirtualMFADevice']['SerialNumber'], AuthenticationCode1='123456', AuthenticationCode2='123456')
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.users) == 1
        assert len(iam.users[0].mfa_devices) == 1
        assert iam.users[0].mfa_devices[0].serial_number == f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:mfa/{mfa_device_name}'
        assert iam.users[0].mfa_devices[0].type == 'mfa'

    @mock_iam
    def test__list_virtual_mfa_devices__(self):
        if False:
            return 10
        iam_client = client('iam')
        username = 'user1'
        iam_client.create_user(UserName=username)
        mfa_device_name = 'test-mfa-device'
        virtual_mfa_device = iam_client.create_virtual_mfa_device(VirtualMFADeviceName=mfa_device_name)
        iam_client.enable_mfa_device(UserName=username, SerialNumber=virtual_mfa_device['VirtualMFADevice']['SerialNumber'], AuthenticationCode1='123456', AuthenticationCode2='123456')
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.virtual_mfa_devices) == 1
        assert iam.virtual_mfa_devices[0]['SerialNumber'] == f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:mfa/{mfa_device_name}'
        assert iam.virtual_mfa_devices[0]['User']['UserName'] == username

    @mock_iam
    def test__get_group_users__(self):
        if False:
            return 10
        iam_client = client('iam')
        username = 'user1'
        iam_client.create_user(UserName=username)
        group = 'test-group'
        iam_client.create_group(GroupName=group)
        iam_client.add_user_to_group(GroupName=group, UserName=username)
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.groups) == 1
        assert iam.groups[0].name == group
        assert iam.groups[0].arn == f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:group/{group}'
        assert len(iam.groups[0].users) == 1
        assert iam.groups[0].users[0].name == username

    @mock_iam
    def test__list_attached_group_policies__(self):
        if False:
            return 10
        iam_client = client('iam')
        username = 'user1'
        iam_client.create_user(UserName=username)
        group = 'test-group'
        iam_client.create_group(GroupName=group)
        policy_document = '\n{\n  "Version": "2012-10-17",\n  "Statement":\n    {\n      "Effect": "Allow",\n      "Action": "s3:ListBucket",\n      "Resource": "arn:aws:s3:::example_bucket"\n    }\n}\n'
        policy_name = 'policy1'
        policy = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=policy_document)
        iam_client.attach_group_policy(GroupName=group, PolicyArn=policy['Policy']['Arn'])
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.groups) == 1
        assert iam.groups[0].name == group
        assert len(iam.groups[0].attached_policies) == 1
        assert iam.groups[0].attached_policies[0]['PolicyName'] == policy_name
        assert iam.groups[0].attached_policies[0]['PolicyArn'] == policy['Policy']['Arn']

    @mock_iam
    def test__list_attached_role_policies__(self):
        if False:
            print('Hello World!')
        iam = client('iam')
        role_name = 'test'
        assume_role_policy_document = {'Version': '2012-10-17', 'Statement': {'Sid': 'test', 'Effect': 'Allow', 'Principal': {'AWS': '*'}, 'Action': 'sts:AssumeRole'}}
        response = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=dumps(assume_role_policy_document))
        iam.attach_role_policy(RoleName=role_name, PolicyArn=READ_ONLY_ACCESS_POLICY_ARN)
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.roles) == 1
        assert iam.roles[0].name == role_name
        assert iam.roles[0].arn == response['Role']['Arn']
        assert len(iam.roles[0].attached_policies) == 1
        assert iam.roles[0].attached_policies[0]['PolicyName'] == 'ReadOnlyAccess'
        assert iam.roles[0].attached_policies[0]['PolicyArn'] == READ_ONLY_ACCESS_POLICY_ARN

    @mock_iam
    def test__get_entities_attached_to_support_roles__no_roles(self):
        if False:
            return 10
        iam_client = client('iam')
        _ = iam_client.list_entities_for_policy(PolicyArn=SUPPORT_SERVICE_ROLE_POLICY_ARN, EntityFilter='Role')['PolicyRoles']
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.entities_role_attached_to_support_policy) == 0

    @mock_iam
    def test__get_entities_attached_to_support_roles__(self):
        if False:
            print('Hello World!')
        iam_client = client('iam')
        role_name = 'test_support'
        assume_role_policy_document = {'Version': '2012-10-17', 'Statement': {'Sid': 'test', 'Effect': 'Allow', 'Principal': {'AWS': '*'}, 'Action': 'sts:AssumeRole'}}
        iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=dumps(assume_role_policy_document))
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=SUPPORT_SERVICE_ROLE_POLICY_ARN)
        iam_client.list_entities_for_policy(PolicyArn=SUPPORT_SERVICE_ROLE_POLICY_ARN, EntityFilter='Role')['PolicyRoles']
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.entities_role_attached_to_support_policy) == 1
        assert iam.entities_role_attached_to_support_policy[0]['RoleName'] == role_name

    @mock_iam
    def test__get_entities_attached_to_securityaudit_roles__no_roles(self):
        if False:
            return 10
        iam_client = client('iam')
        _ = iam_client.list_entities_for_policy(PolicyArn=SECURITY_AUDIT_POLICY_ARN, EntityFilter='Role')['PolicyRoles']
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.entities_role_attached_to_securityaudit_policy) == 0

    @mock_iam
    def test__get_entities_attached_to_securityaudit_roles__(self):
        if False:
            print('Hello World!')
        iam_client = client('iam')
        role_name = 'test_securityaudit'
        assume_role_policy_document = {'Version': '2012-10-17', 'Statement': {'Sid': 'test', 'Effect': 'Allow', 'Principal': {'AWS': '*'}, 'Action': 'sts:AssumeRole'}}
        iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=dumps(assume_role_policy_document))
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=SECURITY_AUDIT_POLICY_ARN)
        iam_client.list_entities_for_policy(PolicyArn=SECURITY_AUDIT_POLICY_ARN, EntityFilter='Role')['PolicyRoles']
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.entities_role_attached_to_securityaudit_policy) == 1
        assert iam.entities_role_attached_to_securityaudit_policy[0]['RoleName'] == role_name

    @mock_iam
    def test___list_policies__(self):
        if False:
            for i in range(10):
                print('nop')
        iam_client = client('iam')
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'logs:CreateLogGroup', 'Resource': '*'}]}
        iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document), Tags=[{'Key': 'string', 'Value': 'string'}])
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        custom_policies = 0
        for policy in iam.policies:
            if policy.type == 'Custom':
                custom_policies += 1
                assert policy.name == 'policy1'
                assert policy.tags == [{'Key': 'string', 'Value': 'string'}]
        assert custom_policies == 1

    @mock_iam
    def test__list_policies_version__(self):
        if False:
            print('Hello World!')
        iam_client = client('iam')
        policy_name = 'policy2'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': '*', 'Resource': '*'}]}
        iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        custom_policies = 0
        for policy in iam.policies:
            if policy.type == 'Custom':
                custom_policies += 1
                assert policy.name == 'policy2'
                assert policy.document['Statement'][0]['Effect'] == 'Allow'
                assert policy.document['Statement'][0]['Action'] == '*'
                assert policy.document['Statement'][0]['Resource'] == '*'
        assert custom_policies == 1

    @mock_iam
    def test__list_saml_providers__(self):
        if False:
            for i in range(10):
                print('nop')
        iam_client = client('iam')
        xml_template = '<EntityDescriptor\n    xmlns="urn:oasis:names:tc:SAML:2.0:metadata"\n    entityID="loadbalancer-9.siroe.com">\n    <SPSSODescriptor\n        AuthnRequestsSigned="false"\n        WantAssertionsSigned="false"\n        protocolSupportEnumeration=\n            "urn:oasis:names:tc:SAML:2.0:protocol">\n        <KeyDescriptor use="signing">\n            <KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#">\n                <X509Data>\n                    <X509Certificate>\nMIICYDCCAgqgAwIBAgICBoowDQYJKoZIhvcNAQEEBQAwgZIxCzAJBgNVBAYTAlVTMRMwEQYDVQQI\nEwpDYWxpZm9ybmlhMRQwEgYDVQQHEwtTYW50YSBDbGFyYTEeMBwGA1UEChMVU3VuIE1pY3Jvc3lz\ndGVtcyBJbmMuMRowGAYDVQQLExFJZGVudGl0eSBTZXJ2aWNlczEcMBoGA1UEAxMTQ2VydGlmaWNh\ndGUgTWFuYWdlcjAeFw0wNjExMDIxOTExMzRaFw0xMDA3MjkxOTExMzRaMDcxEjAQBgNVBAoTCXNp\ncm9lLmNvbTEhMB8GA1UEAxMYbG9hZGJhbGFuY2VyLTkuc2lyb2UuY29tMIGfMA0GCSqGSIb3DQEB\nAQUAA4GNADCBiQKBgQCjOwa5qoaUuVnknqf5pdgAJSEoWlvx/jnUYbkSDpXLzraEiy2UhvwpoBgB\nEeTSUaPPBvboCItchakPI6Z/aFdH3Wmjuij9XD8r1C+q//7sUO0IGn0ORycddHhoo0aSdnnxGf9V\ntREaqKm9dJ7Yn7kQHjo2eryMgYxtr/Z5Il5F+wIDAQABo2AwXjARBglghkgBhvhCAQEEBAMCBkAw\nDgYDVR0PAQH/BAQDAgTwMB8GA1UdIwQYMBaAFDugITflTCfsWyNLTXDl7cMDUKuuMBgGA1UdEQQR\nMA+BDW1hbGxhQHN1bi5jb20wDQYJKoZIhvcNAQEEBQADQQB/6DOB6sRqCZu2OenM9eQR0gube85e\nnTTxU4a7x1naFxzYXK1iQ1vMARKMjDb19QEJIEJKZlDK4uS7yMlf1nFS\n                    </X509Certificate>\n                </X509Data>\n            </KeyInfo>\n        </KeyDescriptor>\n</EntityDescriptor>'
        saml_provider_name = 'test'
        iam_client.create_saml_provider(SAMLMetadataDocument=xml_template, Name=saml_provider_name)
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.saml_providers) == 1
        assert iam.saml_providers[0]['Arn'].split('/')[1] == saml_provider_name

    @mock_iam
    def test__list_inline_user_policies__(self):
        if False:
            print('Hello World!')
        iam_client = client('iam')
        user_name = 'test_user'
        user_arn = iam_client.create_user(UserName=user_name)['User']['Arn']
        policy_name = 'test_not_admin_inline_policy'
        _ = iam_client.put_user_policy(UserName=user_name, PolicyName=policy_name, PolicyDocument=dumps(INLINE_POLICY_NOT_ADMIN))
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.users) == 1
        assert iam.users[0].name == user_name
        assert iam.users[0].arn == user_arn
        assert iam.users[0].mfa_devices == []
        assert iam.users[0].password_last_used is None
        assert iam.users[0].attached_policies == []
        assert iam.users[0].inline_policies == [policy_name]
        assert iam.users[0].tags == []
        for policy in iam.policies:
            if policy.name == policy_name:
                assert policy == Policy(name=policy_name, arn=user_arn, version_id='v1', type='Inline', attached=True, document=INLINE_POLICY_NOT_ADMIN, entity=user_name)

    @mock_iam
    def test__list_inline_group_policies__(self):
        if False:
            i = 10
            return i + 15
        iam_client = client('iam')
        group_name = 'test_group'
        group_arn = iam_client.create_group(GroupName=group_name)['Group']['Arn']
        policy_name = 'test_not_admin_inline_policy'
        _ = iam_client.put_group_policy(GroupName=group_name, PolicyName=policy_name, PolicyDocument=dumps(INLINE_POLICY_NOT_ADMIN))
        iam_client.delete_policy
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.groups) == 1
        assert iam.groups[0].name == group_name
        assert iam.groups[0].arn == group_arn
        assert iam.groups[0].attached_policies == []
        assert iam.groups[0].inline_policies == [policy_name]
        assert iam.groups[0].users == []
        for policy in iam.policies:
            if policy.name == policy_name:
                assert policy == Policy(name=policy_name, arn=group_arn, version_id='v1', type='Inline', attached=True, document=INLINE_POLICY_NOT_ADMIN, entity=group_name)

    @mock_iam
    def test__list_inline_role_policies__(self):
        if False:
            return 10
        iam_client = client('iam')
        role_name = 'test_role'
        role_arn = iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=dumps(ASSUME_ROLE_POLICY_DOCUMENT))['Role']['Arn']
        policy_name = 'test_not_admin_inline_policy'
        _ = iam_client.put_role_policy(RoleName=role_name, PolicyName=policy_name, PolicyDocument=dumps(INLINE_POLICY_NOT_ADMIN))
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.roles) == 1
        assert iam.roles[0].name == role_name
        assert iam.roles[0].arn == role_arn
        assert iam.roles[0].assume_role_policy == ASSUME_ROLE_POLICY_DOCUMENT
        assert not iam.roles[0].is_service_role
        assert iam.roles[0].attached_policies == []
        assert iam.roles[0].inline_policies == [policy_name]
        assert iam.roles[0].tags == []
        for policy in iam.policies:
            if policy.name == policy_name:
                assert policy == Policy(name=policy_name, arn=role_arn, version_id='v1', type='Inline', attached=True, document=INLINE_POLICY_NOT_ADMIN, entity=role_name)

    @mock_iam
    def test__get_user_temporary_credentials_usage__(self):
        if False:
            print('Hello World!')
        iam_client = client('iam')
        username = 'test-user'
        user = iam_client.create_user(UserName=username)
        user_arn = user['User']['Arn']
        access_key = iam_client.create_access_key(UserName='test-user')
        access_key_id = access_key['AccessKey']['AccessKeyId']
        audit_info = self.set_mocked_audit_info()
        iam = IAM(audit_info)
        assert len(iam.users) == 1
        assert len(iam.access_keys_metadata) == 1
        assert iam.access_keys_metadata[username, user_arn]
        assert iam.access_keys_metadata[username, user_arn][0]['UserName'] == username
        assert iam.access_keys_metadata[username, user_arn][0]['AccessKeyId'] == access_key_id
        assert iam.access_keys_metadata[username, user_arn][0]['Status'] == 'Active'
        assert iam.access_keys_metadata[username, user_arn][0]['CreateDate']
        assert iam.last_accessed_services[username, user_arn] == IAM_LAST_ACCESSED_SERVICES
        assert iam.user_temporary_credentials_usage[username, user_arn]