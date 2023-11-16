from re import search
from unittest import mock
import botocore
from boto3 import client, session
from moto import mock_rds
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'us-east-1'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        i = 10
        return i + 15
    if operation_name == 'DescribeDBEngineVersions':
        return {'DBEngineVersions': [{'Engine': 'mysql', 'EngineVersion': '8.0.32', 'DBEngineDescription': 'description', 'DBEngineVersionDescription': 'description'}]}
    return make_api_call(self, operation_name, kwarg)

@mock.patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
class Test_rds_instance_enhanced_monitoring_enabled:

    def set_mocked_audit_info(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None, region_name=AWS_REGION), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=AWS_REGION, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_rds
    def test_rds_no_instances(self):
        if False:
            return 10
        from prowler.providers.aws.services.rds.rds_service import RDS
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.rds.rds_instance_enhanced_monitoring_enabled.rds_instance_enhanced_monitoring_enabled.rds_client', new=RDS(audit_info)):
                from prowler.providers.aws.services.rds.rds_instance_enhanced_monitoring_enabled.rds_instance_enhanced_monitoring_enabled import rds_instance_enhanced_monitoring_enabled
                check = rds_instance_enhanced_monitoring_enabled()
                result = check.execute()
                assert len(result) == 0

    @mock_rds
    def test_rds_instance_no_monitoring(self):
        if False:
            print('Hello World!')
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_instance(DBInstanceIdentifier='db-master-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small')
        from prowler.providers.aws.services.rds.rds_service import RDS
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.rds.rds_instance_enhanced_monitoring_enabled.rds_instance_enhanced_monitoring_enabled.rds_client', new=RDS(audit_info)):
                from prowler.providers.aws.services.rds.rds_instance_enhanced_monitoring_enabled.rds_instance_enhanced_monitoring_enabled import rds_instance_enhanced_monitoring_enabled
                check = rds_instance_enhanced_monitoring_enabled()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'FAIL'
                assert search('does not have enhanced monitoring enabled', result[0].status_extended)
                assert result[0].resource_id == 'db-master-1'
                assert result[0].region == AWS_REGION
                assert result[0].resource_arn == f'arn:aws:rds:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:db:db-master-1'
                assert result[0].resource_tags == []

    @mock_rds
    def test_rds_instance_with_monitoring(self):
        if False:
            i = 10
            return i + 15
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_instance(DBInstanceIdentifier='db-master-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small')
        from prowler.providers.aws.services.rds.rds_service import RDS
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.rds.rds_instance_enhanced_monitoring_enabled.rds_instance_enhanced_monitoring_enabled.rds_client', new=RDS(audit_info)) as service_client:
                from prowler.providers.aws.services.rds.rds_instance_enhanced_monitoring_enabled.rds_instance_enhanced_monitoring_enabled import rds_instance_enhanced_monitoring_enabled
                service_client.db_instances[0].enhanced_monitoring_arn = 'log-stream'
                check = rds_instance_enhanced_monitoring_enabled()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'PASS'
                assert search('has enhanced monitoring enabled', result[0].status_extended)
                assert result[0].resource_id == 'db-master-1'
                assert result[0].region == AWS_REGION
                assert result[0].resource_arn == f'arn:aws:rds:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:db:db-master-1'
                assert result[0].resource_tags == []