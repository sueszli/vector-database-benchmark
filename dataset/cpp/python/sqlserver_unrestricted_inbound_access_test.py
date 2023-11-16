from unittest import mock
from uuid import uuid4

from azure.mgmt.sql.models import FirewallRule

from prowler.providers.azure.services.sqlserver.sqlserver_service import SQL_Server

AZURE_SUSCRIPTION = str(uuid4())


class Test_defender_ensure_defender_for_storage_is_on:
    def test_no_sql_servers(self):
        sqlserver_client = mock.MagicMock
        sqlserver_client.sql_servers = {}

        with mock.patch(
            "prowler.providers.azure.services.sqlserver.sqlserver_unrestricted_inbound_access.sqlserver_unrestricted_inbound_access.sqlserver_client",
            new=sqlserver_client,
        ):
            from prowler.providers.azure.services.sqlserver.sqlserver_unrestricted_inbound_access.sqlserver_unrestricted_inbound_access import (
                sqlserver_unrestricted_inbound_access,
            )

            check = sqlserver_unrestricted_inbound_access()
            result = check.execute()
            assert len(result) == 0

    def test_sql_servers_unrestricted_inbound_access(self):
        sqlserver_client = mock.MagicMock
        sql_server_name = "SQL Server Name"
        sql_server_id = str(uuid4())
        sqlserver_client.sql_servers = {
            AZURE_SUSCRIPTION: [
                SQL_Server(
                    id=sql_server_id,
                    name=sql_server_name,
                    public_network_access="",
                    minimal_tls_version="",
                    administrators=None,
                    auditing_policies=[],
                    firewall_rules=[
                        FirewallRule(
                            start_ip_address="0.0.0.0", end_ip_address="255.255.255.255"
                        )
                    ],
                )
            ]
        }

        with mock.patch(
            "prowler.providers.azure.services.sqlserver.sqlserver_unrestricted_inbound_access.sqlserver_unrestricted_inbound_access.sqlserver_client",
            new=sqlserver_client,
        ):
            from prowler.providers.azure.services.sqlserver.sqlserver_unrestricted_inbound_access.sqlserver_unrestricted_inbound_access import (
                sqlserver_unrestricted_inbound_access,
            )

            check = sqlserver_unrestricted_inbound_access()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "FAIL"
            assert (
                result[0].status_extended
                == f"SQL Server {sql_server_name} from subscription {AZURE_SUSCRIPTION} has firewall rules allowing 0.0.0.0-255.255.255.255."
            )
            assert result[0].subscription == AZURE_SUSCRIPTION
            assert result[0].resource_name == sql_server_name
            assert result[0].resource_id == sql_server_id

    def test_sql_servers_restricted_inbound_access(self):
        sqlserver_client = mock.MagicMock
        sql_server_name = "SQL Server Name"
        sql_server_id = str(uuid4())
        sqlserver_client.sql_servers = {
            AZURE_SUSCRIPTION: [
                SQL_Server(
                    id=sql_server_id,
                    name=sql_server_name,
                    public_network_access="",
                    minimal_tls_version="",
                    administrators=None,
                    auditing_policies=[],
                    firewall_rules=[
                        FirewallRule(
                            start_ip_address="10.10.10.10", end_ip_address="10.10.10.10"
                        )
                    ],
                )
            ]
        }

        with mock.patch(
            "prowler.providers.azure.services.sqlserver.sqlserver_unrestricted_inbound_access.sqlserver_unrestricted_inbound_access.sqlserver_client",
            new=sqlserver_client,
        ):
            from prowler.providers.azure.services.sqlserver.sqlserver_unrestricted_inbound_access.sqlserver_unrestricted_inbound_access import (
                sqlserver_unrestricted_inbound_access,
            )

            check = sqlserver_unrestricted_inbound_access()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "PASS"
            assert (
                result[0].status_extended
                == f"SQL Server {sql_server_name} from subscription {AZURE_SUSCRIPTION} does not have firewall rules allowing 0.0.0.0-255.255.255.255."
            )
            assert result[0].subscription == AZURE_SUSCRIPTION
            assert result[0].resource_name == sql_server_name
            assert result[0].resource_id == sql_server_id
