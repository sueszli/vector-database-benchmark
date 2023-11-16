from re import search
from unittest import mock
from uuid import uuid4
from prowler.providers.aws.services.redshift.redshift_service import Cluster
AWS_REGION = 'eu-west-1'
AWS_ACCOUNT_NUMBER = '123456789012'
CLUSTER_ID = str(uuid4())
CLUSTER_ARN = f'arn:aws:redshift:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:cluster:{CLUSTER_ID}'

class Test_redshift_cluster_audit_logging:

    def test_no_clusters(self):
        if False:
            i = 10
            return i + 15
        redshift_client = mock.MagicMock
        redshift_client.clusters = []
        with mock.patch('prowler.providers.aws.services.redshift.redshift_service.Redshift', redshift_client):
            from prowler.providers.aws.services.redshift.redshift_cluster_audit_logging.redshift_cluster_audit_logging import redshift_cluster_audit_logging
            check = redshift_cluster_audit_logging()
            result = check.execute()
            assert len(result) == 0

    def test_cluster_is_not_audit_logging(self):
        if False:
            while True:
                i = 10
        redshift_client = mock.MagicMock
        redshift_client.clusters = []
        redshift_client.clusters.append(Cluster(id=CLUSTER_ID, arn=CLUSTER_ARN, region=AWS_REGION, logging_enabled=False))
        with mock.patch('prowler.providers.aws.services.redshift.redshift_service.Redshift', redshift_client):
            from prowler.providers.aws.services.redshift.redshift_cluster_audit_logging.redshift_cluster_audit_logging import redshift_cluster_audit_logging
            check = redshift_cluster_audit_logging()
            result = check.execute()
            assert result[0].status == 'FAIL'
            assert search('has audit logging disabled', result[0].status_extended)
            assert result[0].resource_id == CLUSTER_ID
            assert result[0].resource_arn == CLUSTER_ARN

    def test_cluster_is_audit_logging(self):
        if False:
            i = 10
            return i + 15
        redshift_client = mock.MagicMock
        redshift_client.clusters = []
        redshift_client.clusters.append(Cluster(id=CLUSTER_ID, arn=CLUSTER_ARN, region=AWS_REGION, logging_enabled=True, endpoint_address='192.192.192.192'))
        with mock.patch('prowler.providers.aws.services.redshift.redshift_service.Redshift', redshift_client):
            from prowler.providers.aws.services.redshift.redshift_cluster_audit_logging.redshift_cluster_audit_logging import redshift_cluster_audit_logging
            check = redshift_cluster_audit_logging()
            result = check.execute()
            assert result[0].status == 'PASS'
            assert search('has audit logging enabled', result[0].status_extended)
            assert result[0].resource_id == CLUSTER_ID
            assert result[0].resource_arn == CLUSTER_ARN