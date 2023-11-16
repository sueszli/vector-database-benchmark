from unittest import mock
from uuid import uuid4
from prowler.providers.azure.services.storage.storage_service import Storage_Account
AZURE_SUSCRIPTION = str(uuid4())

class Test_storage_infrastructure_encryption_is_enabled:

    def test_storage_no_storage_accounts(self):
        if False:
            for i in range(10):
                print('nop')
        storage_client = mock.MagicMock
        storage_client.storage_accounts = {}
        with mock.patch('prowler.providers.azure.services.storage.storage_infrastructure_encryption_is_enabled.storage_infrastructure_encryption_is_enabled.storage_client', new=storage_client):
            from prowler.providers.azure.services.storage.storage_infrastructure_encryption_is_enabled.storage_infrastructure_encryption_is_enabled import storage_infrastructure_encryption_is_enabled
            check = storage_infrastructure_encryption_is_enabled()
            result = check.execute()
            assert len(result) == 0

    def test_storage_storage_accounts_infrastructure_encryption_disabled(self):
        if False:
            i = 10
            return i + 15
        storage_account_id = str(uuid4())
        storage_account_name = 'Test Storage Account'
        storage_client = mock.MagicMock
        storage_client.storage_accounts = {AZURE_SUSCRIPTION: [Storage_Account(id=storage_account_id, name=storage_account_name, enable_https_traffic_only=False, infrastructure_encryption=False, allow_blob_public_access=None, network_rule_set=None, encryption_type='None', minimum_tls_version='TLS1_1')]}
        with mock.patch('prowler.providers.azure.services.storage.storage_infrastructure_encryption_is_enabled.storage_infrastructure_encryption_is_enabled.storage_client', new=storage_client):
            from prowler.providers.azure.services.storage.storage_infrastructure_encryption_is_enabled.storage_infrastructure_encryption_is_enabled import storage_infrastructure_encryption_is_enabled
            check = storage_infrastructure_encryption_is_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Storage account {storage_account_name} from subscription {AZURE_SUSCRIPTION} has infrastructure encryption disabled.'
            assert result[0].subscription == AZURE_SUSCRIPTION
            assert result[0].resource_name == storage_account_name
            assert result[0].resource_id == storage_account_id

    def test_storage_storage_accounts_infrastructure_encryption_enabled(self):
        if False:
            while True:
                i = 10
        storage_account_id = str(uuid4())
        storage_account_name = 'Test Storage Account'
        storage_client = mock.MagicMock
        storage_client.storage_accounts = {AZURE_SUSCRIPTION: [Storage_Account(id=storage_account_id, name=storage_account_name, enable_https_traffic_only=False, infrastructure_encryption=True, allow_blob_public_access=None, network_rule_set=None, encryption_type='None', minimum_tls_version='TLS1_1')]}
        with mock.patch('prowler.providers.azure.services.storage.storage_infrastructure_encryption_is_enabled.storage_infrastructure_encryption_is_enabled.storage_client', new=storage_client):
            from prowler.providers.azure.services.storage.storage_infrastructure_encryption_is_enabled.storage_infrastructure_encryption_is_enabled import storage_infrastructure_encryption_is_enabled
            check = storage_infrastructure_encryption_is_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Storage account {storage_account_name} from subscription {AZURE_SUSCRIPTION} has infrastructure encryption enabled.'
            assert result[0].subscription == AZURE_SUSCRIPTION
            assert result[0].resource_name == storage_account_name
            assert result[0].resource_id == storage_account_id