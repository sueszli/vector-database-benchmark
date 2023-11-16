from unittest import mock
from uuid import uuid4

from prowler.providers.azure.services.storage.storage_service import Storage_Account

AZURE_SUSCRIPTION = str(uuid4())


class Test_storage_ensure_minimum_tls_version_12:
    def test_storage_no_storage_accounts(self):
        storage_client = mock.MagicMock
        storage_client.storage_accounts = {}

        with mock.patch(
            "prowler.providers.azure.services.storage.storage_ensure_minimum_tls_version_12.storage_ensure_minimum_tls_version_12.storage_client",
            new=storage_client,
        ):
            from prowler.providers.azure.services.storage.storage_ensure_minimum_tls_version_12.storage_ensure_minimum_tls_version_12 import (
                storage_ensure_minimum_tls_version_12,
            )

            check = storage_ensure_minimum_tls_version_12()
            result = check.execute()
            assert len(result) == 0

    def test_storage_storage_accounts_tls_not_1_2(self):
        storage_account_id = str(uuid4())
        storage_account_name = "Test Storage Account"
        storage_client = mock.MagicMock
        storage_client.storage_accounts = {
            AZURE_SUSCRIPTION: [
                Storage_Account(
                    id=storage_account_id,
                    name=storage_account_name,
                    enable_https_traffic_only=False,
                    infrastructure_encryption=False,
                    allow_blob_public_access=None,
                    network_rule_set=None,
                    encryption_type="None",
                    minimum_tls_version="TLS1_1",
                )
            ]
        }

        with mock.patch(
            "prowler.providers.azure.services.storage.storage_ensure_minimum_tls_version_12.storage_ensure_minimum_tls_version_12.storage_client",
            new=storage_client,
        ):
            from prowler.providers.azure.services.storage.storage_ensure_minimum_tls_version_12.storage_ensure_minimum_tls_version_12 import (
                storage_ensure_minimum_tls_version_12,
            )

            check = storage_ensure_minimum_tls_version_12()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "FAIL"
            assert (
                result[0].status_extended
                == f"Storage account {storage_account_name} from subscription {AZURE_SUSCRIPTION} does not have TLS version set to 1.2."
            )
            assert result[0].subscription == AZURE_SUSCRIPTION
            assert result[0].resource_name == storage_account_name
            assert result[0].resource_id == storage_account_id

    def test_storage_storage_accounts_tls_1_2(self):
        storage_account_id = str(uuid4())
        storage_account_name = "Test Storage Account"
        storage_client = mock.MagicMock
        storage_client.storage_accounts = {
            AZURE_SUSCRIPTION: [
                Storage_Account(
                    id=storage_account_id,
                    name=storage_account_name,
                    enable_https_traffic_only=False,
                    infrastructure_encryption=False,
                    allow_blob_public_access=None,
                    network_rule_set=None,
                    encryption_type="None",
                    minimum_tls_version="TLS1_2",
                )
            ]
        }

        with mock.patch(
            "prowler.providers.azure.services.storage.storage_ensure_minimum_tls_version_12.storage_ensure_minimum_tls_version_12.storage_client",
            new=storage_client,
        ):
            from prowler.providers.azure.services.storage.storage_ensure_minimum_tls_version_12.storage_ensure_minimum_tls_version_12 import (
                storage_ensure_minimum_tls_version_12,
            )

            check = storage_ensure_minimum_tls_version_12()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "PASS"
            assert (
                result[0].status_extended
                == f"Storage account {storage_account_name} from subscription {AZURE_SUSCRIPTION} has TLS version set to 1.2."
            )
            assert result[0].subscription == AZURE_SUSCRIPTION
            assert result[0].resource_name == storage_account_name
            assert result[0].resource_id == storage_account_id
