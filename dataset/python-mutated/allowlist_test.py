import yaml
from boto3 import resource, session
from mock import MagicMock
from moto import mock_dynamodb, mock_s3
from prowler.providers.aws.lib.allowlist.allowlist import allowlist_findings, is_allowlisted, is_allowlisted_in_check, is_allowlisted_in_region, is_allowlisted_in_resource, is_allowlisted_in_tags, is_excepted, parse_allowlist_file
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
from tests.providers.aws.audit_info_utils import AWS_ACCOUNT_NUMBER, AWS_REGION_EU_WEST_1, AWS_REGION_US_EAST_1

class Test_Allowlist:

    def set_mocked_audit_info(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_s3
    def test_s3_allowlist(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        s3_resource = resource('s3', region_name=AWS_REGION_US_EAST_1)
        s3_resource.create_bucket(Bucket='test-allowlist')
        s3_resource.Object('test-allowlist', 'allowlist.yaml').put(Body=open('tests/providers/aws/lib/allowlist/fixtures/allowlist.yaml', 'rb'))
        with open('tests/providers/aws/lib/allowlist/fixtures/allowlist.yaml') as f:
            assert yaml.safe_load(f)['Allowlist'] == parse_allowlist_file(audit_info, 's3://test-allowlist/allowlist.yaml')

    @mock_dynamodb
    def test_dynamo_allowlist(self):
        if False:
            return 10
        audit_info = self.set_mocked_audit_info()
        dynamodb_resource = resource('dynamodb', region_name=AWS_REGION_US_EAST_1)
        table_name = 'test-allowlist'
        params = {'TableName': table_name, 'KeySchema': [{'AttributeName': 'Accounts', 'KeyType': 'HASH'}, {'AttributeName': 'Checks', 'KeyType': 'RANGE'}], 'AttributeDefinitions': [{'AttributeName': 'Accounts', 'AttributeType': 'S'}, {'AttributeName': 'Checks', 'AttributeType': 'S'}], 'ProvisionedThroughput': {'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10}}
        table = dynamodb_resource.create_table(**params)
        table.put_item(Item={'Accounts': '*', 'Checks': 'iam_user_hardware_mfa_enabled', 'Regions': [AWS_REGION_EU_WEST_1, AWS_REGION_US_EAST_1], 'Resources': ['keyword']})
        assert 'keyword' in parse_allowlist_file(audit_info, 'arn:aws:dynamodb:' + AWS_REGION_US_EAST_1 + ':' + str(AWS_ACCOUNT_NUMBER) + ':table/' + table_name)['Accounts']['*']['Checks']['iam_user_hardware_mfa_enabled']['Resources']

    @mock_dynamodb
    def test_dynamo_allowlist_with_tags(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        dynamodb_resource = resource('dynamodb', region_name=AWS_REGION_US_EAST_1)
        table_name = 'test-allowlist'
        params = {'TableName': table_name, 'KeySchema': [{'AttributeName': 'Accounts', 'KeyType': 'HASH'}, {'AttributeName': 'Checks', 'KeyType': 'RANGE'}], 'AttributeDefinitions': [{'AttributeName': 'Accounts', 'AttributeType': 'S'}, {'AttributeName': 'Checks', 'AttributeType': 'S'}], 'ProvisionedThroughput': {'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10}}
        table = dynamodb_resource.create_table(**params)
        table.put_item(Item={'Accounts': '*', 'Checks': '*', 'Regions': ['*'], 'Resources': ['*'], 'Tags': ['environment=dev']})
        assert 'environment=dev' in parse_allowlist_file(audit_info, 'arn:aws:dynamodb:' + AWS_REGION_US_EAST_1 + ':' + str(AWS_ACCOUNT_NUMBER) + ':table/' + table_name)['Accounts']['*']['Checks']['*']['Tags']

    def test_allowlist_findings(self):
        if False:
            return 10
        allowlist = {'Accounts': {'*': {'Checks': {'check_test': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['prowler', '^test', 'prowler-pro']}}}}}
        check_findings = []
        finding_1 = MagicMock
        finding_1.check_metadata = MagicMock
        finding_1.check_metadata.CheckID = 'check_test'
        finding_1.status = 'FAIL'
        finding_1.region = AWS_REGION_US_EAST_1
        finding_1.resource_id = 'prowler'
        finding_1.resource_tags = []
        check_findings.append(finding_1)
        allowlisted_findings = allowlist_findings(allowlist, AWS_ACCOUNT_NUMBER, check_findings)
        assert len(allowlisted_findings) == 1
        assert allowlisted_findings[0].status == 'WARNING'

    def test_is_allowlisted_with_everything_excepted(self):
        if False:
            while True:
                i = 10
        allowlist = {'Accounts': {'*': {'Checks': {'athena_*': {'Regions': '*', 'Resources': '*', 'Tags': '*', 'Exceptions': {'Accounts': ['*'], 'Regions': ['*'], 'Resources': ['*'], 'Tags': ['*']}}}}}}
        assert not is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'athena_1', AWS_REGION_US_EAST_1, 'prowler', '')

    def test_is_allowlisted_with_default_allowlist(self):
        if False:
            i = 10
            return i + 15
        allowlist = {'Accounts': {'*': {'Checks': {'*': {'Tags': ['*'], 'Regions': ['*'], 'Resources': ['*'], 'Exceptions': {'Tags': [], 'Regions': [], 'Accounts': [], 'Resources': []}}}}}}
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'athena_1', AWS_REGION_US_EAST_1, 'prowler', '')

    def test_is_allowlisted(self):
        if False:
            i = 10
            return i + 15
        allowlist = {'Accounts': {'*': {'Checks': {'check_test': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['prowler', '^test', 'prowler-pro']}}}}}
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler-test', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'test-prowler', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler-pro-test', '')
        assert not is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', 'us-east-2', 'test', '')

    def test_is_allowlisted_wildcard(self):
        if False:
            for i in range(10):
                print('nop')
        allowlist = {'Accounts': {'*': {'Checks': {'check_test': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['.*']}}}}}
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler-test', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'test-prowler', '')
        assert not is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', 'us-east-2', 'test', '')

    def test_is_allowlisted_asterisk(self):
        if False:
            while True:
                i = 10
        allowlist = {'Accounts': {'*': {'Checks': {'check_test': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['*']}}}}}
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler-test', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'test-prowler', '')
        assert not is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', 'us-east-2', 'test', '')

    def test_is_allowlisted_all_and_single_account(self):
        if False:
            for i in range(10):
                print('nop')
        allowlist = {'Accounts': {'*': {'Checks': {'check_test_2': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['*']}}}, AWS_ACCOUNT_NUMBER: {'Checks': {'check_test': {'Regions': [AWS_REGION_US_EAST_1], 'Resources': ['*']}}}}}
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test_2', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler-test', '')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'test-prowler', '')
        assert not is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', 'us-east-2', 'test', '')

    def test_is_allowlisted_single_account(self):
        if False:
            while True:
                i = 10
        allowlist = {'Accounts': {AWS_ACCOUNT_NUMBER: {'Checks': {'check_test': {'Regions': [AWS_REGION_US_EAST_1], 'Resources': ['prowler']}}}}}
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler', '')
        assert not is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', 'us-east-2', 'test', '')

    def test_is_allowlisted_in_region(self):
        if False:
            return 10
        allowlisted_regions = [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1]
        finding_region = AWS_REGION_US_EAST_1
        assert is_allowlisted_in_region(allowlisted_regions, finding_region)

    def test_is_allowlisted_in_region_wildcard(self):
        if False:
            return 10
        allowlisted_regions = ['*']
        finding_region = AWS_REGION_US_EAST_1
        assert is_allowlisted_in_region(allowlisted_regions, finding_region)

    def test_is_not_allowlisted_in_region(self):
        if False:
            while True:
                i = 10
        allowlisted_regions = [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1]
        finding_region = 'eu-west-2'
        assert not is_allowlisted_in_region(allowlisted_regions, finding_region)

    def test_is_allowlisted_in_check(self):
        if False:
            i = 10
            return i + 15
        allowlisted_checks = {'check_test': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['*']}}
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler-test', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'test-prowler', '')
        assert not is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'check_test', 'us-east-2', 'test', '')

    def test_is_allowlisted_in_check_regex(self):
        if False:
            print('Hello World!')
        allowlisted_checks = {'s3_*': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['*']}}
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 's3_bucket_public_access', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 's3_bucket_no_mfa_delete', AWS_REGION_US_EAST_1, 'prowler-test', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 's3_bucket_policy_public_write_access', AWS_REGION_US_EAST_1, 'test-prowler', '')
        assert not is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'iam_user_hardware_mfa_enabled', AWS_REGION_US_EAST_1, 'test', '')

    def test_is_allowlisted_lambda_generic_check(self):
        if False:
            return 10
        allowlisted_checks = {'lambda_*': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['*']}}
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'awslambda_function_invoke_api_operations_cloudtrail_logging_enabled', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'awslambda_function_no_secrets_in_code', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'awslambda_function_no_secrets_in_variables', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'awslambda_function_not_publicly_accessible', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'awslambda_function_url_cors_policy', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'awslambda_function_url_public', AWS_REGION_US_EAST_1, 'prowler', '')
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'awslambda_function_using_supported_runtimes', AWS_REGION_US_EAST_1, 'prowler', '')

    def test_is_allowlisted_lambda_concrete_check(self):
        if False:
            i = 10
            return i + 15
        allowlisted_checks = {'lambda_function_no_secrets_in_variables': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['*']}}
        assert is_allowlisted_in_check(allowlisted_checks, AWS_ACCOUNT_NUMBER, 'awslambda_function_no_secrets_in_variables', AWS_REGION_US_EAST_1, 'prowler', '')

    def test_is_allowlisted_tags(self):
        if False:
            return 10
        allowlist = {'Accounts': {'*': {'Checks': {'check_test': {'Regions': [AWS_REGION_US_EAST_1, AWS_REGION_EU_WEST_1], 'Resources': ['*'], 'Tags': ['environment=dev', 'project=.*']}}}}}
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler', 'environment=dev')
        assert is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', AWS_REGION_US_EAST_1, 'prowler-test', 'environment=dev | project=prowler')
        assert not is_allowlisted(allowlist, AWS_ACCOUNT_NUMBER, 'check_test', 'us-east-2', 'test', 'environment=pro')

    def test_is_allowlisted_in_tags(self):
        if False:
            while True:
                i = 10
        allowlist_tags = ['environment=dev', 'project=prowler']
        assert is_allowlisted_in_tags(allowlist_tags, 'environment=dev')
        assert is_allowlisted_in_tags(allowlist_tags, 'environment=dev | project=prowler')
        assert not is_allowlisted_in_tags(allowlist_tags, 'environment=pro')

    def test_is_allowlisted_in_tags_regex(self):
        if False:
            print('Hello World!')
        allowlist_tags = ['environment=(dev|test)', '.*=prowler']
        assert is_allowlisted_in_tags(allowlist_tags, 'environment=test | proj=prowler')
        assert is_allowlisted_in_tags(allowlist_tags, 'env=prod | project=prowler')
        assert not is_allowlisted_in_tags(allowlist_tags, 'environment=prod | project=myproj')

    def test_is_allowlisted_in_tags_with_no_tags_in_finding(self):
        if False:
            while True:
                i = 10
        allowlist_tags = ['environment=(dev|test)', '.*=prowler']
        finding_tags = ''
        assert not is_allowlisted_in_tags(allowlist_tags, finding_tags)

    def test_is_excepted(self):
        if False:
            for i in range(10):
                print('nop')
        exceptions = {'Accounts': [AWS_ACCOUNT_NUMBER], 'Regions': ['eu-central-1', 'eu-south-3'], 'Resources': ['test'], 'Tags': ['environment=test', 'project=.*']}
        assert is_excepted(exceptions, AWS_ACCOUNT_NUMBER, 'eu-central-1', 'test', 'environment=test')
        assert is_excepted(exceptions, AWS_ACCOUNT_NUMBER, 'eu-south-3', 'test', 'environment=test')
        assert is_excepted(exceptions, AWS_ACCOUNT_NUMBER, 'eu-south-3', 'test123', 'environment=test')

    def test_is_excepted_all_wildcard(self):
        if False:
            i = 10
            return i + 15
        exceptions = {'Accounts': ['*'], 'Regions': ['*'], 'Resources': ['*'], 'Tags': ['*']}
        assert is_excepted(exceptions, AWS_ACCOUNT_NUMBER, 'eu-south-2', 'test', 'environment=test')
        assert not is_excepted(exceptions, AWS_ACCOUNT_NUMBER, 'eu-south-2', 'test', None)

    def test_is_not_excepted(self):
        if False:
            print('Hello World!')
        exceptions = {'Accounts': [AWS_ACCOUNT_NUMBER], 'Regions': ['eu-central-1', 'eu-south-3'], 'Resources': ['test'], 'Tags': ['environment=test', 'project=.*']}
        assert not is_excepted(exceptions, AWS_ACCOUNT_NUMBER, 'eu-south-2', 'test', 'environment=test')
        assert not is_excepted(exceptions, AWS_ACCOUNT_NUMBER, 'eu-south-3', 'prowler', 'environment=test')
        assert not is_excepted(exceptions, AWS_ACCOUNT_NUMBER, 'eu-south-3', 'test', 'environment=pro')

    def test_is_allowlisted_in_resource(self):
        if False:
            print('Hello World!')
        allowlist_resources = ['prowler', '^test', 'prowler-pro']
        assert is_allowlisted_in_resource(allowlist_resources, 'prowler')
        assert is_allowlisted_in_resource(allowlist_resources, 'prowler-test')
        assert is_allowlisted_in_resource(allowlist_resources, 'test-prowler')
        assert not is_allowlisted_in_resource(allowlist_resources, 'random')