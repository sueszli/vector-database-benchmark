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
        print('Hello World!')
    if operation_name == 'DescribeDBEngineVersions':
        return {'DBEngineVersions': [{'Engine': 'mysql', 'EngineVersion': '8.0.32', 'DBEngineDescription': 'description', 'DBEngineVersionDescription': 'description'}]}
    return make_api_call(self, operation_name, kwarg)

@mock.patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
class Test_rds_instance_deletion_protection:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None, region_name=AWS_REGION), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=AWS_REGION, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_rds
    def test_rds_no_instances(self):
        if False:
            print('Hello World!')
        from prowler.providers.aws.services.rds.rds_service import RDS
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection.rds_client', new=RDS(audit_info)):
                from prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection import rds_instance_deletion_protection
                check = rds_instance_deletion_protection()
                result = check.execute()
                assert len(result) == 0

    @mock_rds
    def test_rds_instance_no_deletion_protection(self):
        if False:
            print('Hello World!')
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_instance(DBInstanceIdentifier='db-master-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small')
        from prowler.providers.aws.services.rds.rds_service import RDS
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection.rds_client', new=RDS(audit_info)):
                from prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection import rds_instance_deletion_protection
                check = rds_instance_deletion_protection()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'FAIL'
                assert search('deletion protection is not enabled', result[0].status_extended)
                assert result[0].resource_id == 'db-master-1'
                assert result[0].region == AWS_REGION
                assert result[0].resource_arn == f'arn:aws:rds:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:db:db-master-1'
                assert result[0].resource_tags == []

    @mock_rds
    def test_rds_instance_with_deletion_protection(self):
        if False:
            print('Hello World!')
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_instance(DBInstanceIdentifier='db-master-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small', DeletionProtection=True)
        from prowler.providers.aws.services.rds.rds_service import RDS
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection.rds_client', new=RDS(audit_info)):
                from prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection import rds_instance_deletion_protection
                check = rds_instance_deletion_protection()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'PASS'
                assert search('deletion protection is enabled', result[0].status_extended)
                assert result[0].resource_id == 'db-master-1'
                assert result[0].region == AWS_REGION
                assert result[0].resource_arn == f'arn:aws:rds:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:db:db-master-1'
                assert result[0].resource_tags == []

    @mock_rds
    def test_rds_instance_without_cluster_deletion_protection(self):
        if False:
            while True:
                i = 10
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_cluster(DBClusterIdentifier='db-cluster-1', AllocatedStorage=10, Engine='postgres', DatabaseName='staging-postgres', DeletionProtection=False, MasterUsername='test', MasterUserPassword='password', Tags=[{'Key': 'test', 'Value': 'test'}])
        conn.create_db_instance(DBInstanceIdentifier='db-master-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small', DBClusterIdentifier='db-cluster-1')
        from prowler.providers.aws.services.rds.rds_service import RDS
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection.rds_client', new=RDS(audit_info)):
                from prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection import rds_instance_deletion_protection
                check = rds_instance_deletion_protection()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'FAIL'
                assert search('deletion protection is not enabled at cluster', result[0].status_extended)
                assert result[0].resource_id == 'db-master-1'
                assert result[0].region == AWS_REGION
                assert result[0].resource_arn == f'arn:aws:rds:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:db:db-master-1'
                assert result[0].resource_tags == []

    @mock_rds
    def test_rds_instance_with_cluster_deletion_protection(self):
        if False:
            for i in range(10):
                print('nop')
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_cluster(DBClusterIdentifier='db-cluster-1', AllocatedStorage=10, Engine='postgres', DatabaseName='staging-postgres', DeletionProtection=True, MasterUsername='test', MasterUserPassword='password', Tags=[{'Key': 'test', 'Value': 'test'}])
        conn.create_db_instance(DBInstanceIdentifier='db-master-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small', DBClusterIdentifier='db-cluster-1')
        from prowler.providers.aws.services.rds.rds_service import RDS
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection.rds_client', new=RDS(audit_info)):
                from prowler.providers.aws.services.rds.rds_instance_deletion_protection.rds_instance_deletion_protection import rds_instance_deletion_protection
                check = rds_instance_deletion_protection()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'PASS'
                assert search('deletion protection is enabled at cluster', result[0].status_extended)
                assert result[0].resource_id == 'db-master-1'
                assert result[0].region == AWS_REGION
                assert result[0].resource_arn == f'arn:aws:rds:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:db:db-master-1'
                assert result[0].resource_tags == []