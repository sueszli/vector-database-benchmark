import botocore
from boto3 import session
from mock import patch
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.elasticache.elasticache_service import Cluster, ElastiCache
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_ACCOUNT_ARN = f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'
AWS_REGION = 'us-east-1'
AWS_REGION_AZ1 = 'us-east-1a'
AWS_REGION_AZ2 = 'us-east-b'
SUBNET_GROUP_NAME = 'default'
SUBNET_1 = 'subnet-1'
SUBNET_2 = 'subnet-2'
ELASTICACHE_CLUSTER_NAME = 'test-cluster'
ELASTICACHE_CLUSTER_ARN = f'arn:aws:elasticache:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:{ELASTICACHE_CLUSTER_NAME}'
ELASTICACHE_ENGINE = 'redis'
ELASTICACHE_CLUSTER_TAGS = [{'Key': 'environment', 'Value': 'test'}]
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwargs):
    if False:
        i = 10
        return i + 15
    '\n    As you can see the operation_name has the list_analyzers snake_case form but\n    we are using the ListAnalyzers form.\n    Rationale -> https://github.com/boto/botocore/blob/develop/botocore/client.py#L810:L816\n\n    We have to mock every AWS API call using Boto3\n    '
    if operation_name == 'DescribeCacheClusters':
        return {'CacheClusters': [{'CacheClusterId': ELASTICACHE_CLUSTER_NAME, 'CacheSubnetGroupName': SUBNET_GROUP_NAME, 'ARN': ELASTICACHE_CLUSTER_ARN}]}
    if operation_name == 'DescribeCacheSubnetGroups':
        return {'CacheSubnetGroups': [{'CacheSubnetGroupName': SUBNET_GROUP_NAME, 'CacheSubnetGroupDescription': 'Subnet Group', 'VpcId': 'vpc-1', 'SubnetGroupStatus': 'Complete', 'Subnets': [{'SubnetIdentifier': 'subnet-1', 'SubnetAvailabilityZone': {'Name': AWS_REGION_AZ1}, 'SubnetStatus': 'Active'}, {'SubnetIdentifier': 'subnet-2', 'SubnetAvailabilityZone': {'Name': AWS_REGION_AZ2}, 'SubnetStatus': 'Active'}], 'DBSubnetGroupArn': f'arn:aws:rds:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:subgrp:{SUBNET_GROUP_NAME}'}]}
    if operation_name == 'ListTagsForResource':
        return {'TagList': ELASTICACHE_CLUSTER_TAGS}
    return make_api_call(self, operation_name, kwargs)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        while True:
            i = 10
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
class Test_ElastiCache_Service:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=AWS_ACCOUNT_ARN, audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test_service(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        elasticache = ElastiCache(audit_info)
        assert elasticache.service == 'elasticache'

    def test_client(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        elasticache = ElastiCache(audit_info)
        assert elasticache.client.__class__.__name__ == 'ElastiCache'

    def test__get_session__(self):
        if False:
            return 10
        audit_info = self.set_mocked_audit_info()
        elasticache = ElastiCache(audit_info)
        assert elasticache.session.__class__.__name__ == 'Session'

    def test_audited_account(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        elasticache = ElastiCache(audit_info)
        assert elasticache.audited_account == AWS_ACCOUNT_NUMBER

    def test_describe_cache_clusters(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        elasticache = ElastiCache(audit_info)
        assert len(elasticache.clusters) == 1
        assert elasticache.clusters[ELASTICACHE_CLUSTER_ARN]
        assert elasticache.clusters[ELASTICACHE_CLUSTER_ARN] == Cluster(arn=ELASTICACHE_CLUSTER_ARN, name=ELASTICACHE_CLUSTER_NAME, id=ELASTICACHE_CLUSTER_NAME, region=AWS_REGION, cache_subnet_group_id=SUBNET_GROUP_NAME, subnets=[SUBNET_1, SUBNET_2], tags=ELASTICACHE_CLUSTER_TAGS)