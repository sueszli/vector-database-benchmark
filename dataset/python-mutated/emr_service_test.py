from datetime import datetime
from unittest.mock import patch
import botocore
from boto3 import client, session
from moto import mock_emr
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.emr.emr_service import EMR, ClusterStatus
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        print('Hello World!')
    'We have to mock every AWS API call using Boto3'
    if operation_name == 'GetBlockPublicAccessConfiguration':
        return {'BlockPublicAccessConfiguration': {'BlockPublicSecurityGroupRules': True, 'PermittedPublicSecurityGroupRuleRanges': [{'MinRange': 0, 'MaxRange': 65535}]}, 'BlockPublicAccessConfigurationMetadata': {'CreationDateTime': datetime(2015, 1, 1), 'CreatedByArn': 'test-arn'}}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        print('Hello World!')
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
class Test_EMR_Service:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=DEFAULT_ACCOUNT_ID, audited_account_arn=f'arn:aws:iam::{DEFAULT_ACCOUNT_ID}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_emr
    def test__get_client__(self):
        if False:
            for i in range(10):
                print('nop')
        emr = EMR(self.set_mocked_audit_info())
        assert emr.regional_clients[AWS_REGION].__class__.__name__ == 'EMR'

    @mock_emr
    def test__get_session__(self):
        if False:
            return 10
        emr = EMR(self.set_mocked_audit_info())
        assert emr.session.__class__.__name__ == 'Session'

    @mock_emr
    def test__get_service__(self):
        if False:
            for i in range(10):
                print('nop')
        emr = EMR(self.set_mocked_audit_info())
        assert emr.service == 'emr'

    @mock_emr
    def test__list_clusters__(self):
        if False:
            return 10
        emr_client = client('emr', region_name=AWS_REGION)
        cluster_name = 'test-cluster'
        run_job_flow_args = dict(Instances={'InstanceCount': 3, 'KeepJobFlowAliveWhenNoSteps': True, 'MasterInstanceType': 'c3.medium', 'Placement': {'AvailabilityZone': 'us-east-1a'}, 'SlaveInstanceType': 'c3.xlarge'}, JobFlowRole='EMR_EC2_DefaultRole', LogUri='s3://mybucket/log', Name=cluster_name, ServiceRole='EMR_DefaultRole', VisibleToAllUsers=True, Tags=[{'Key': 'test', 'Value': 'test'}])
        cluster_id = emr_client.run_job_flow(**run_job_flow_args)['JobFlowId']
        emr = EMR(self.set_mocked_audit_info())
        assert len(emr.clusters) == 1
        assert emr.clusters[cluster_id].id == cluster_id
        assert emr.clusters[cluster_id].name == cluster_name
        assert emr.clusters[cluster_id].status == ClusterStatus.WAITING
        assert emr.clusters[cluster_id].arn == f'arn:aws:elasticmapreduce:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:cluster/{cluster_id}'
        assert emr.clusters[cluster_id].region == AWS_REGION
        assert emr.clusters[cluster_id].master_public_dns_name == 'ec2-184-0-0-1.us-west-1.compute.amazonaws.com'
        assert emr.clusters[cluster_id].public
        assert emr.clusters[cluster_id].tags == [{'Key': 'test', 'Value': 'test'}]

    @mock_emr
    def test__get_block_public_access_configuration__(self):
        if False:
            while True:
                i = 10
        emr = EMR(self.set_mocked_audit_info())
        assert len(emr.block_public_access_configuration) == 1
        assert emr.block_public_access_configuration[AWS_REGION].block_public_security_group_rules