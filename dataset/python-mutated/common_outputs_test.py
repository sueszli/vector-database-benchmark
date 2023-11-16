from argparse import Namespace
from os import rmdir
from boto3 import session
from mock import patch
from prowler.lib.outputs.html import get_assessment_summary
from prowler.providers.aws.lib.audit_info.audit_info import AWS_Audit_Info
from prowler.providers.azure.lib.audit_info.audit_info import Azure_Audit_Info, Azure_Identity_Info, Azure_Region_Config
from prowler.providers.common.models import Audit_Metadata
from prowler.providers.common.outputs import Aws_Output_Options, Azure_Output_Options, Gcp_Output_Options, set_provider_output_options
from prowler.providers.gcp.lib.audit_info.models import GCP_Audit_Info
AWS_ACCOUNT_NUMBER = '012345678912'
DATETIME = '20230101120000'

@patch('prowler.providers.common.outputs.output_file_timestamp', new=DATETIME)
class Test_Common_Output_Options:

    def set_mocked_azure_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = Azure_Audit_Info(credentials=None, identity=Azure_Identity_Info(), audit_metadata=None, audit_resources=None, audit_config=None, azure_region_config=Azure_Region_Config())
        return audit_info

    def set_mocked_gcp_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = GCP_Audit_Info(credentials=None, default_project_id='test-project1', project_ids=['test-project1', 'test-project2'], audit_resources=None, audit_metadata=None, audit_config=None)
        return audit_info

    def set_mocked_aws_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id='test-user', audited_partition='aws', audited_identity_arn='test-user-arn', profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test_set_provider_output_options_aws(self):
        if False:
            for i in range(10):
                print('nop')
        provider = 'aws'
        arguments = Namespace()
        arguments.quiet = True
        arguments.output_modes = ['html', 'csv', 'json']
        arguments.output_directory = 'output_test_directory'
        arguments.verbose = True
        arguments.output_filename = 'output_test_filename'
        arguments.security_hub = True
        arguments.shodan = 'test-api-key'
        arguments.only_logs = False
        arguments.unix_timestamp = False
        audit_info = self.set_mocked_aws_audit_info()
        allowlist_file = ''
        bulk_checks_metadata = {}
        output_options = set_provider_output_options(provider, arguments, audit_info, allowlist_file, bulk_checks_metadata)
        assert isinstance(output_options, Aws_Output_Options)
        assert output_options.security_hub_enabled
        assert output_options.is_quiet
        assert output_options.output_modes == ['html', 'csv', 'json', 'json-asff']
        assert output_options.output_directory == arguments.output_directory
        assert output_options.allowlist_file == ''
        assert output_options.bulk_checks_metadata == {}
        assert output_options.verbose
        assert output_options.output_filename == arguments.output_filename
        rmdir(arguments.output_directory)

    def test_set_provider_output_options_gcp(self):
        if False:
            return 10
        provider = 'gcp'
        arguments = Namespace()
        arguments.quiet = True
        arguments.output_modes = ['html', 'csv', 'json']
        arguments.output_directory = 'output_test_directory'
        arguments.verbose = True
        arguments.output_filename = 'output_test_filename'
        arguments.only_logs = False
        arguments.unix_timestamp = False
        audit_info = self.set_mocked_gcp_audit_info()
        allowlist_file = ''
        bulk_checks_metadata = {}
        output_options = set_provider_output_options(provider, arguments, audit_info, allowlist_file, bulk_checks_metadata)
        assert isinstance(output_options, Gcp_Output_Options)
        assert output_options.is_quiet
        assert output_options.output_modes == ['html', 'csv', 'json']
        assert output_options.output_directory == arguments.output_directory
        assert output_options.allowlist_file == ''
        assert output_options.bulk_checks_metadata == {}
        assert output_options.verbose
        assert output_options.output_filename == arguments.output_filename
        rmdir(arguments.output_directory)

    def test_set_provider_output_options_aws_no_output_filename(self):
        if False:
            for i in range(10):
                print('nop')
        provider = 'aws'
        arguments = Namespace()
        arguments.quiet = True
        arguments.output_modes = ['html', 'csv', 'json']
        arguments.output_directory = 'output_test_directory'
        arguments.verbose = True
        arguments.security_hub = True
        arguments.shodan = 'test-api-key'
        arguments.only_logs = False
        arguments.unix_timestamp = False
        audit_info = self.set_mocked_aws_audit_info()
        allowlist_file = ''
        bulk_checks_metadata = {}
        output_options = set_provider_output_options(provider, arguments, audit_info, allowlist_file, bulk_checks_metadata)
        assert isinstance(output_options, Aws_Output_Options)
        assert output_options.security_hub_enabled
        assert output_options.is_quiet
        assert output_options.output_modes == ['html', 'csv', 'json', 'json-asff']
        assert output_options.output_directory == arguments.output_directory
        assert output_options.allowlist_file == ''
        assert output_options.bulk_checks_metadata == {}
        assert output_options.verbose
        assert output_options.output_filename == f'prowler-output-{AWS_ACCOUNT_NUMBER}-{DATETIME}'
        rmdir(arguments.output_directory)

    def test_set_provider_output_options_azure_domain(self):
        if False:
            i = 10
            return i + 15
        provider = 'azure'
        arguments = Namespace()
        arguments.quiet = True
        arguments.output_modes = ['html', 'csv', 'json']
        arguments.output_directory = 'output_test_directory'
        arguments.verbose = True
        arguments.only_logs = False
        arguments.unix_timestamp = False
        audit_info = self.set_mocked_azure_audit_info()
        audit_info.identity.domain = 'test-domain'
        allowlist_file = ''
        bulk_checks_metadata = {}
        output_options = set_provider_output_options(provider, arguments, audit_info, allowlist_file, bulk_checks_metadata)
        assert isinstance(output_options, Azure_Output_Options)
        assert output_options.is_quiet
        assert output_options.output_modes == ['html', 'csv', 'json']
        assert output_options.output_directory == arguments.output_directory
        assert output_options.allowlist_file == ''
        assert output_options.bulk_checks_metadata == {}
        assert output_options.verbose
        assert output_options.output_filename == f'prowler-output-{audit_info.identity.domain}-{DATETIME}'
        rmdir(arguments.output_directory)

    def test_set_provider_output_options_azure_tenant_ids(self):
        if False:
            while True:
                i = 10
        provider = 'azure'
        arguments = Namespace()
        arguments.quiet = True
        arguments.output_modes = ['html', 'csv', 'json']
        arguments.output_directory = 'output_test_directory'
        arguments.verbose = True
        arguments.only_logs = False
        arguments.unix_timestamp = False
        audit_info = self.set_mocked_azure_audit_info()
        tenants = ['tenant-1', 'tenant-2']
        audit_info.identity.tenant_ids = tenants
        allowlist_file = ''
        bulk_checks_metadata = {}
        output_options = set_provider_output_options(provider, arguments, audit_info, allowlist_file, bulk_checks_metadata)
        assert isinstance(output_options, Azure_Output_Options)
        assert output_options.is_quiet
        assert output_options.output_modes == ['html', 'csv', 'json']
        assert output_options.output_directory == arguments.output_directory
        assert output_options.allowlist_file == ''
        assert output_options.bulk_checks_metadata == {}
        assert output_options.verbose
        assert output_options.output_filename == f"prowler-output-{'-'.join(tenants)}-{DATETIME}"
        rmdir(arguments.output_directory)

    def test_azure_get_assessment_summary(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_azure_audit_info()
        tenants = ['tenant-1', 'tenant-2']
        audit_info.identity.tenant_ids = tenants
        audit_info.identity.subscriptions = {'Azure subscription 1': '12345-qwerty', 'Subscription2': '12345-qwerty'}
        printed_subscriptions = []
        for (key, value) in audit_info.identity.subscriptions.items():
            intermediate = key + ' : ' + value
            printed_subscriptions.append(intermediate)
        assert get_assessment_summary(audit_info) == f"""\n            <div class="col-md-2">\n                <div class="card">\n                    <div class="card-header">\n                        Azure Assessment Summary\n                    </div>\n                    <ul class="list-group list-group-flush">\n                        <li class="list-group-item">\n                            <b>Azure Tenant IDs:</b> {' '.join(audit_info.identity.tenant_ids)}\n                        </li>\n                        <li class="list-group-item">\n                            <b>Azure Tenant Domain:</b> {audit_info.identity.domain}\n                        </li>\n                        <li class="list-group-item">\n                            <b>Azure Subscriptions:</b> {' '.join(printed_subscriptions)}\n                        </li>\n                    </ul>\n                </div>\n            </div>\n            <div class="col-md-4">\n            <div class="card">\n                <div class="card-header">\n                    Azure Credentials\n                </div>\n                <ul class="list-group list-group-flush">\n                    <li class="list-group-item">\n                        <b>Azure Identity Type:</b> {audit_info.identity.identity_type}\n                        </li>\n                        <li class="list-group-item">\n                            <b>Azure Identity ID:</b> {audit_info.identity.identity_id}\n                        </li>\n                    </ul>\n                </div>\n            </div>\n            """

    def test_aws_get_assessment_summary(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_aws_audit_info()
        assert get_assessment_summary(audit_info) == f'\n            <div class="col-md-2">\n                <div class="card">\n                    <div class="card-header">\n                        AWS Assessment Summary\n                    </div>\n                    <ul class="list-group list-group-flush">\n                        <li class="list-group-item">\n                            <b>AWS Account:</b> {audit_info.audited_account}\n                        </li>\n                        <li class="list-group-item">\n                            <b>AWS-CLI Profile:</b> {audit_info.profile}\n                        </li>\n                        <li class="list-group-item">\n                            <b>Audited Regions:</b> All Regions\n                        </li>\n                    </ul>\n                </div>\n            </div>\n            <div class="col-md-4">\n            <div class="card">\n                <div class="card-header">\n                    AWS Credentials\n                </div>\n                <ul class="list-group list-group-flush">\n                    <li class="list-group-item">\n                        <b>User Id:</b> {audit_info.audited_user_id}\n                        </li>\n                        <li class="list-group-item">\n                            <b>Caller Identity ARN:</b> {audit_info.audited_identity_arn}\n                        </li>\n                    </ul>\n                </div>\n            </div>\n            '

    def test_gcp_get_assessment_summary(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_gcp_audit_info()
        profile = 'default'
        assert get_assessment_summary(audit_info) == f"""\n            <div class="col-md-2">\n                <div class="card">\n                    <div class="card-header">\n                        GCP Assessment Summary\n                    </div>\n                    <ul class="list-group list-group-flush">\n                        <li class="list-group-item">\n                            <b>GCP Project IDs:</b> {', '.join(audit_info.project_ids)}\n                        </li>\n                    </ul>\n                </div>\n            </div>\n            <div class="col-md-4">\n                <div class="card">\n                    <div class="card-header">\n                        GCP Credentials\n                    </div>\n                    <ul class="list-group list-group-flush">\n                        <li class="list-group-item">\n                            <b>GCP Account:</b> {profile}\n                        </li>\n                    </ul>\n                </div>\n            </div>\n            """