from os import path
import botocore
from boto3 import session
from mock import MagicMock, patch
from prowler.config.config import prowler_version, timestamp_utc
from prowler.lib.check.models import Check_Report, load_check_metadata
from prowler.providers.aws.lib.security_hub.security_hub import batch_send_to_security_hub, prepare_security_hub_findings, verify_security_hub_integration_enabled_per_region
from tests.providers.aws.audit_info_utils import AWS_ACCOUNT_NUMBER, AWS_REGION_EU_WEST_1, AWS_REGION_EU_WEST_2, set_mocked_aws_audit_info
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        return 10
    if operation_name == 'BatchImportFindings':
        return {'FailedCount': 0, 'SuccessCount': 1}
    if operation_name == 'DescribeHub':
        return {'HubArn': f'arn:aws:securityhub:{AWS_REGION_EU_WEST_1}:{AWS_ACCOUNT_NUMBER}:hub/default', 'SubscribedAt': '2023-02-07T09:45:43.742Z', 'AutoEnableControls': True, 'ControlFindingGenerator': 'STANDARD_CONTROL'}
    if operation_name == 'ListEnabledProductsForImport':
        return {'ProductSubscriptions': [f'arn:aws:securityhub:{AWS_REGION_EU_WEST_1}:{AWS_ACCOUNT_NUMBER}:product-subscription/prowler/prowler']}
    return make_api_call(self, operation_name, kwarg)

class Test_SecurityHub:

    def generate_finding(self, status, region):
        if False:
            return 10
        finding = Check_Report(load_check_metadata(f'{path.dirname(path.realpath(__file__))}/fixtures/metadata.json').json())
        finding.status = status
        finding.status_extended = 'test'
        finding.resource_id = 'test'
        finding.resource_arn = 'test'
        finding.region = region
        return finding

    def set_mocked_output_options(self, is_quiet):
        if False:
            while True:
                i = 10
        output_options = MagicMock
        output_options.bulk_checks_metadata = {}
        output_options.is_quiet = is_quiet
        return output_options

    def set_mocked_session(self, region):
        if False:
            while True:
                i = 10
        return session.Session(region_name=region)

    @patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
    def test_verify_security_hub_integration_enabled_per_region(self):
        if False:
            print('Hello World!')
        session = self.set_mocked_session(AWS_REGION_EU_WEST_1)
        assert verify_security_hub_integration_enabled_per_region(AWS_REGION_EU_WEST_1, session, AWS_ACCOUNT_NUMBER)

    def test_prepare_security_hub_findings_enabled_region_not_quiet(self):
        if False:
            while True:
                i = 10
        enabled_regions = [AWS_REGION_EU_WEST_1]
        output_options = self.set_mocked_output_options(is_quiet=False)
        findings = [self.generate_finding('PASS', AWS_REGION_EU_WEST_1)]
        audit_info = set_mocked_aws_audit_info(audited_regions=[AWS_REGION_EU_WEST_1, AWS_REGION_EU_WEST_2])
        assert prepare_security_hub_findings(findings, audit_info, output_options, enabled_regions) == {AWS_REGION_EU_WEST_1: [{'SchemaVersion': '2018-10-08', 'Id': f'prowler-iam_user_accesskey_unused-{AWS_ACCOUNT_NUMBER}-{AWS_REGION_EU_WEST_1}-ee26b0dd4', 'ProductArn': f'arn:aws:securityhub:{AWS_REGION_EU_WEST_1}::product/prowler/prowler', 'RecordState': 'ACTIVE', 'ProductFields': {'ProviderName': 'Prowler', 'ProviderVersion': prowler_version, 'ProwlerResourceName': 'test'}, 'GeneratorId': 'prowler-iam_user_accesskey_unused', 'AwsAccountId': f'{AWS_ACCOUNT_NUMBER}', 'Types': ['Software and Configuration Checks'], 'FirstObservedAt': timestamp_utc.strftime('%Y-%m-%dT%H:%M:%SZ'), 'UpdatedAt': timestamp_utc.strftime('%Y-%m-%dT%H:%M:%SZ'), 'CreatedAt': timestamp_utc.strftime('%Y-%m-%dT%H:%M:%SZ'), 'Severity': {'Label': 'LOW'}, 'Title': 'Ensure Access Keys unused are disabled', 'Description': 'test', 'Resources': [{'Type': 'AwsIamAccessAnalyzer', 'Id': 'test', 'Partition': 'aws', 'Region': f'{AWS_REGION_EU_WEST_1}'}], 'Compliance': {'Status': 'PASSED', 'RelatedRequirements': [], 'AssociatedStandards': []}, 'Remediation': {'Recommendation': {'Text': 'Run sudo yum update and cross your fingers and toes.', 'Url': 'https://myfp.com/recommendations/dangerous_things_and_how_to_fix_them.html'}}}]}

    def test_prepare_security_hub_findings_quiet_INFO_finding(self):
        if False:
            print('Hello World!')
        enabled_regions = [AWS_REGION_EU_WEST_1]
        output_options = self.set_mocked_output_options(is_quiet=False)
        findings = [self.generate_finding('INFO', AWS_REGION_EU_WEST_1)]
        audit_info = set_mocked_aws_audit_info(audited_regions=[AWS_REGION_EU_WEST_1, AWS_REGION_EU_WEST_2])
        assert prepare_security_hub_findings(findings, audit_info, output_options, enabled_regions) == {AWS_REGION_EU_WEST_1: []}

    def test_prepare_security_hub_findings_disabled_region(self):
        if False:
            print('Hello World!')
        enabled_regions = [AWS_REGION_EU_WEST_1]
        output_options = self.set_mocked_output_options(is_quiet=False)
        findings = [self.generate_finding('PASS', AWS_REGION_EU_WEST_2)]
        audit_info = set_mocked_aws_audit_info(audited_regions=[AWS_REGION_EU_WEST_1, AWS_REGION_EU_WEST_2])
        assert prepare_security_hub_findings(findings, audit_info, output_options, enabled_regions) == {AWS_REGION_EU_WEST_1: []}

    def test_prepare_security_hub_findings_quiet(self):
        if False:
            i = 10
            return i + 15
        enabled_regions = [AWS_REGION_EU_WEST_1]
        output_options = self.set_mocked_output_options(is_quiet=True)
        findings = [self.generate_finding('PASS', AWS_REGION_EU_WEST_1)]
        audit_info = set_mocked_aws_audit_info(audited_regions=[AWS_REGION_EU_WEST_1, AWS_REGION_EU_WEST_2])
        assert prepare_security_hub_findings(findings, audit_info, output_options, enabled_regions) == {AWS_REGION_EU_WEST_1: []}

    def test_prepare_security_hub_findings_no_audited_regions(self):
        if False:
            for i in range(10):
                print('nop')
        enabled_regions = [AWS_REGION_EU_WEST_1]
        output_options = self.set_mocked_output_options(is_quiet=False)
        findings = [self.generate_finding('PASS', AWS_REGION_EU_WEST_1)]
        audit_info = set_mocked_aws_audit_info()
        assert prepare_security_hub_findings(findings, audit_info, output_options, enabled_regions) == {AWS_REGION_EU_WEST_1: [{'SchemaVersion': '2018-10-08', 'Id': f'prowler-iam_user_accesskey_unused-{AWS_ACCOUNT_NUMBER}-{AWS_REGION_EU_WEST_1}-ee26b0dd4', 'ProductArn': f'arn:aws:securityhub:{AWS_REGION_EU_WEST_1}::product/prowler/prowler', 'RecordState': 'ACTIVE', 'ProductFields': {'ProviderName': 'Prowler', 'ProviderVersion': prowler_version, 'ProwlerResourceName': 'test'}, 'GeneratorId': 'prowler-iam_user_accesskey_unused', 'AwsAccountId': f'{AWS_ACCOUNT_NUMBER}', 'Types': ['Software and Configuration Checks'], 'FirstObservedAt': timestamp_utc.strftime('%Y-%m-%dT%H:%M:%SZ'), 'UpdatedAt': timestamp_utc.strftime('%Y-%m-%dT%H:%M:%SZ'), 'CreatedAt': timestamp_utc.strftime('%Y-%m-%dT%H:%M:%SZ'), 'Severity': {'Label': 'LOW'}, 'Title': 'Ensure Access Keys unused are disabled', 'Description': 'test', 'Resources': [{'Type': 'AwsIamAccessAnalyzer', 'Id': 'test', 'Partition': 'aws', 'Region': f'{AWS_REGION_EU_WEST_1}'}], 'Compliance': {'Status': 'PASSED', 'RelatedRequirements': [], 'AssociatedStandards': []}, 'Remediation': {'Recommendation': {'Text': 'Run sudo yum update and cross your fingers and toes.', 'Url': 'https://myfp.com/recommendations/dangerous_things_and_how_to_fix_them.html'}}}]}

    @patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
    def test_batch_send_to_security_hub_one_finding(self):
        if False:
            print('Hello World!')
        enabled_regions = [AWS_REGION_EU_WEST_1]
        output_options = self.set_mocked_output_options(is_quiet=False)
        findings = [self.generate_finding('PASS', AWS_REGION_EU_WEST_1)]
        audit_info = set_mocked_aws_audit_info(audited_regions=[AWS_REGION_EU_WEST_1, AWS_REGION_EU_WEST_2])
        session = self.set_mocked_session(AWS_REGION_EU_WEST_1)
        security_hub_findings = prepare_security_hub_findings(findings, audit_info, output_options, enabled_regions)
        assert batch_send_to_security_hub(security_hub_findings, session) == 1