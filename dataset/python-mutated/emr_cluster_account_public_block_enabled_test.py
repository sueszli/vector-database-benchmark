from unittest import mock
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.services.emr.emr_service import BlockPublicAccessConfiguration
AWS_REGION = 'eu-west-1'

class Test_emr_cluster_account_public_block_enabled:

    def test_account_public_block_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        emr_client = mock.MagicMock
        emr_client.audited_account = DEFAULT_ACCOUNT_ID
        emr_client.block_public_access_configuration = {AWS_REGION: BlockPublicAccessConfiguration(block_public_security_group_rules=True)}
        with mock.patch('prowler.providers.aws.services.emr.emr_service.EMR', new=emr_client):
            from prowler.providers.aws.services.emr.emr_cluster_account_public_block_enabled.emr_cluster_account_public_block_enabled import emr_cluster_account_public_block_enabled
            check = emr_cluster_account_public_block_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == AWS_REGION
            assert result[0].resource_id == DEFAULT_ACCOUNT_ID
            assert result[0].status == 'PASS'
            assert result[0].status_extended == 'EMR Account has Block Public Access enabled.'

    def test_account_public_block_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        emr_client = mock.MagicMock
        emr_client.audited_account = DEFAULT_ACCOUNT_ID
        emr_client.block_public_access_configuration = {AWS_REGION: BlockPublicAccessConfiguration(block_public_security_group_rules=False)}
        with mock.patch('prowler.providers.aws.services.emr.emr_service.EMR', new=emr_client):
            from prowler.providers.aws.services.emr.emr_cluster_account_public_block_enabled.emr_cluster_account_public_block_enabled import emr_cluster_account_public_block_enabled
            check = emr_cluster_account_public_block_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == AWS_REGION
            assert result[0].resource_id == DEFAULT_ACCOUNT_ID
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == 'EMR Account has Block Public Access disabled.'