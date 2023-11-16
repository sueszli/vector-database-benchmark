from re import search
from unittest import mock
from uuid import uuid4
from prowler.providers.aws.services.redshift.redshift_service import Cluster
AWS_REGION = 'eu-west-1'
AWS_ACCOUNT_NUMBER = '123456789012'
CLUSTER_ID = str(uuid4())
CLUSTER_ARN = f'arn:aws:redshift:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:cluster:{CLUSTER_ID}'

class Test_redshift_cluster_automatic_upgrades:

    def test_no_clusters(self):
        if False:
            print('Hello World!')
        redshift_client = mock.MagicMock
        redshift_client.clusters = []
        with mock.patch('prowler.providers.aws.services.redshift.redshift_service.Redshift', redshift_client):
            from prowler.providers.aws.services.redshift.redshift_cluster_automatic_upgrades.redshift_cluster_automatic_upgrades import redshift_cluster_automatic_upgrades
            check = redshift_cluster_automatic_upgrades()
            result = check.execute()
            assert len(result) == 0

    def test_cluster_not_automatic_upgrades(self):
        if False:
            return 10
        redshift_client = mock.MagicMock
        redshift_client.clusters = []
        redshift_client.clusters.append(Cluster(id=CLUSTER_ID, arn=CLUSTER_ARN, region=AWS_REGION, allow_version_upgrade=False))
        with mock.patch('prowler.providers.aws.services.redshift.redshift_service.Redshift', redshift_client):
            from prowler.providers.aws.services.redshift.redshift_cluster_automatic_upgrades.redshift_cluster_automatic_upgrades import redshift_cluster_automatic_upgrades
            check = redshift_cluster_automatic_upgrades()
            result = check.execute()
            assert result[0].status == 'FAIL'
            assert search('has AllowVersionUpgrade disabled', result[0].status_extended)
            assert result[0].resource_id == CLUSTER_ID
            assert result[0].resource_arn == CLUSTER_ARN

    def test_cluster_automatic_upgrades(self):
        if False:
            for i in range(10):
                print('nop')
        redshift_client = mock.MagicMock
        redshift_client.clusters = []
        redshift_client.clusters.append(Cluster(id=CLUSTER_ID, arn=CLUSTER_ARN, region=AWS_REGION, allow_version_upgrade=True))
        with mock.patch('prowler.providers.aws.services.redshift.redshift_service.Redshift', redshift_client):
            from prowler.providers.aws.services.redshift.redshift_cluster_automatic_upgrades.redshift_cluster_automatic_upgrades import redshift_cluster_automatic_upgrades
            check = redshift_cluster_automatic_upgrades()
            result = check.execute()
            assert result[0].status == 'PASS'
            assert search('has AllowVersionUpgrade enabled', result[0].status_extended)
            assert result[0].resource_id == CLUSTER_ID
            assert result[0].resource_arn == CLUSTER_ARN