from datetime import datetime
from unittest.mock import patch
import botocore
from boto3 import session
from prowler.providers.aws.lib.audit_info.audit_info import AWS_Audit_Info
from prowler.providers.aws.services.backup.backup_service import Backup
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        print('Hello World!')
    '\n    Mock every AWS API call\n    '
    if operation_name == 'ListBackupVaults':
        return {'BackupVaultList': [{'BackupVaultArn': 'ARN', 'BackupVaultName': 'Test Vault', 'EncryptionKeyArn': '', 'NumberOfRecoveryPoints': 0, 'Locked': True, 'MinRetentionDays': 1, 'MaxRetentionDays': 2}]}
    if operation_name == 'ListBackupPlans':
        return {'BackupPlansList': [{'BackupPlanArn': 'ARN', 'BackupPlanId': 'ID', 'BackupPlanName': 'Test Plan', 'VersionId': 'test_version_id', 'LastExecutionDate': datetime(2015, 1, 1), 'AdvancedBackupSettings': []}]}
    if operation_name == 'ListReportPlans':
        return {'ReportPlans': [{'ReportPlanArn': 'ARN', 'ReportPlanName': 'Test Report Plan', 'LastAttemptedExecutionTime': datetime(2015, 1, 1), 'LastSuccessfulExecutionTime': datetime(2015, 1, 1)}]}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        print('Hello World!')
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_Backup_Service:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=None, audited_account_arn=None, audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test__get_client__(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        backup = Backup(audit_info)
        assert backup.regional_clients[AWS_REGION].__class__.__name__ == 'Backup'

    def test__get_session__(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        access_analyzer = Backup(audit_info)
        assert access_analyzer.session.__class__.__name__ == 'Session'

    def test__get_service__(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        access_analyzer = Backup(audit_info)
        assert access_analyzer.service == 'backup'

    def test__list_backup_vaults__(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        backup = Backup(audit_info)
        assert len(backup.backup_vaults) == 1
        assert backup.backup_vaults[0].arn == 'ARN'
        assert backup.backup_vaults[0].name == 'Test Vault'
        assert backup.backup_vaults[0].region == AWS_REGION
        assert backup.backup_vaults[0].encryption == ''
        assert backup.backup_vaults[0].recovery_points == 0
        assert backup.backup_vaults[0].locked is True
        assert backup.backup_vaults[0].min_retention_days == 1
        assert backup.backup_vaults[0].max_retention_days == 2

    def test__list_backup_plans__(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        backup = Backup(audit_info)
        assert len(backup.backup_plans) == 1
        assert backup.backup_plans[0].arn == 'ARN'
        assert backup.backup_plans[0].id == 'ID'
        assert backup.backup_plans[0].region == AWS_REGION
        assert backup.backup_plans[0].name == 'Test Plan'
        assert backup.backup_plans[0].version_id == 'test_version_id'
        assert backup.backup_plans[0].last_execution_date == datetime(2015, 1, 1)
        assert backup.backup_plans[0].advanced_settings == []

    def test__list_backup_report_plans__(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        backup = Backup(audit_info)
        assert len(backup.backup_report_plans) == 1
        assert backup.backup_report_plans[0].arn == 'ARN'
        assert backup.backup_report_plans[0].region == AWS_REGION
        assert backup.backup_report_plans[0].name == 'Test Report Plan'
        assert backup.backup_report_plans[0].last_attempted_execution_date == datetime(2015, 1, 1)
        assert backup.backup_report_plans[0].last_successful_execution_date == datetime(2015, 1, 1)