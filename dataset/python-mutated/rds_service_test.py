from unittest.mock import patch
import botocore
from boto3 import client, session
from moto import mock_rds
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.rds.rds_service import RDS
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

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
class Test_RDS_Service:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=[AWS_REGION], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_rds
    def test_service(self):
        if False:
            return 10
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        assert rds.service == 'rds'

    @mock_rds
    def test_client(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        for regional_client in rds.regional_clients.values():
            assert regional_client.__class__.__name__ == 'RDS'

    @mock_rds
    def test__get_session__(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        assert rds.session.__class__.__name__ == 'Session'

    @mock_rds
    def test_audited_account(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        assert rds.audited_account == AWS_ACCOUNT_NUMBER

    @mock_rds
    def test__describe_db_instances__(self):
        if False:
            print('Hello World!')
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_parameter_group(DBParameterGroupName='test', DBParameterGroupFamily='default.postgres9.3', Description='test parameter group')
        conn.create_db_instance(DBInstanceIdentifier='db-master-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small', StorageEncrypted=True, DeletionProtection=True, PubliclyAccessible=True, AutoMinorVersionUpgrade=True, BackupRetentionPeriod=10, EnableCloudwatchLogsExports=['audit', 'error'], MultiAZ=True, DBParameterGroupName='test', Tags=[{'Key': 'test', 'Value': 'test'}])
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        assert len(rds.db_instances) == 1
        assert rds.db_instances[0].id == 'db-master-1'
        assert rds.db_instances[0].region == AWS_REGION
        assert rds.db_instances[0].endpoint['Address'] == 'db-master-1.aaaaaaaaaa.us-east-1.rds.amazonaws.com'
        assert rds.db_instances[0].status == 'available'
        assert rds.db_instances[0].public
        assert rds.db_instances[0].encrypted
        assert rds.db_instances[0].backup_retention_period == 10
        assert rds.db_instances[0].cloudwatch_logs == ['audit', 'error']
        assert rds.db_instances[0].deletion_protection
        assert rds.db_instances[0].auto_minor_version_upgrade
        assert rds.db_instances[0].multi_az
        assert rds.db_instances[0].tags == [{'Key': 'test', 'Value': 'test'}]
        assert 'test' in rds.db_instances[0].parameter_groups

    @mock_rds
    def test__describe_db_parameters__(self):
        if False:
            i = 10
            return i + 15
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_parameter_group(DBParameterGroupName='test', DBParameterGroupFamily='default.postgres9.3', Description='test parameter group')
        conn.create_db_instance(DBInstanceIdentifier='db-master-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small', DBParameterGroupName='test')
        conn.modify_db_parameter_group(DBParameterGroupName='test', Parameters=[{'ParameterName': 'rds.force_ssl', 'ParameterValue': '1', 'ApplyMethod': 'immediate'}])
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        assert len(rds.db_instances) == 1
        assert rds.db_instances[0].id == 'db-master-1'
        assert rds.db_instances[0].region == AWS_REGION
        for parameter in rds.db_instances[0].parameters:
            if parameter['ParameterName'] == 'rds.force_ssl':
                assert parameter['ParameterValue'] == '1'

    @mock_rds
    def test__describe_db_snapshots__(self):
        if False:
            for i in range(10):
                print('nop')
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_instance(DBInstanceIdentifier='db-primary-1', AllocatedStorage=10, Engine='postgres', DBName='staging-postgres', DBInstanceClass='db.m1.small')
        conn.create_db_snapshot(DBInstanceIdentifier='db-primary-1', DBSnapshotIdentifier='snapshot-1')
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        assert len(rds.db_snapshots) == 1
        assert rds.db_snapshots[0].id == 'snapshot-1'
        assert rds.db_snapshots[0].instance_id == 'db-primary-1'
        assert rds.db_snapshots[0].region == AWS_REGION
        assert not rds.db_snapshots[0].public

    @mock_rds
    def test__describe_db_clusters__(self):
        if False:
            print('Hello World!')
        conn = client('rds', region_name=AWS_REGION)
        cluster_id = 'db-master-1'
        conn.create_db_parameter_group(DBParameterGroupName='test', DBParameterGroupFamily='default.postgres9.3', Description='test parameter group')
        conn.create_db_cluster(DBClusterIdentifier=cluster_id, AllocatedStorage=10, Engine='postgres', DatabaseName='staging-postgres', StorageEncrypted=False, DeletionProtection=True, PubliclyAccessible=False, AutoMinorVersionUpgrade=False, BackupRetentionPeriod=1, MasterUsername='test', MasterUserPassword='password', EnableCloudwatchLogsExports=['audit', 'error'], DBClusterParameterGroupName='test', Tags=[{'Key': 'test', 'Value': 'test'}])
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        db_cluster_arn = f'arn:aws:rds:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:cluster:{cluster_id}'
        assert len(rds.db_clusters) == 1
        assert rds.db_clusters[db_cluster_arn].id == 'db-master-1'
        assert rds.db_clusters[db_cluster_arn].engine == 'postgres'
        assert rds.db_clusters[db_cluster_arn].region == AWS_REGION
        assert f'{AWS_REGION}.rds.amazonaws.com' in rds.db_clusters[db_cluster_arn].endpoint
        assert rds.db_clusters[db_cluster_arn].status == 'available'
        assert not rds.db_clusters[db_cluster_arn].public
        assert not rds.db_clusters[db_cluster_arn].encrypted
        assert rds.db_clusters[db_cluster_arn].backup_retention_period == 1
        assert rds.db_clusters[db_cluster_arn].cloudwatch_logs == ['audit', 'error']
        assert rds.db_clusters[db_cluster_arn].deletion_protection
        assert not rds.db_clusters[db_cluster_arn].auto_minor_version_upgrade
        assert not rds.db_clusters[db_cluster_arn].multi_az
        assert rds.db_clusters[db_cluster_arn].tags == [{'Key': 'test', 'Value': 'test'}]
        assert rds.db_clusters[db_cluster_arn].parameter_group == 'test'

    @mock_rds
    def test__describe_db_cluster_snapshots__(self):
        if False:
            for i in range(10):
                print('nop')
        conn = client('rds', region_name=AWS_REGION)
        conn.create_db_cluster(DBClusterIdentifier='db-primary-1', AllocatedStorage=10, Engine='postgres', DBClusterInstanceClass='db.m1.small', MasterUsername='root', MasterUserPassword='hunter2000')
        conn.create_db_cluster_snapshot(DBClusterIdentifier='db-primary-1', DBClusterSnapshotIdentifier='snapshot-1')
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        assert len(rds.db_cluster_snapshots) == 1
        assert rds.db_cluster_snapshots[0].id == 'snapshot-1'
        assert rds.db_cluster_snapshots[0].cluster_id == 'db-primary-1'
        assert rds.db_cluster_snapshots[0].region == AWS_REGION
        assert not rds.db_cluster_snapshots[0].public

    @mock_rds
    def test__describe_db_engine_versions__(self):
        if False:
            return 10
        audit_info = self.set_mocked_audit_info()
        rds = RDS(audit_info)
        assert 'mysql' in rds.db_engines[AWS_REGION]
        assert rds.db_engines[AWS_REGION]['mysql'].engine_versions == ['8.0.32']
        assert rds.db_engines[AWS_REGION]['mysql'].engine_description == 'description'