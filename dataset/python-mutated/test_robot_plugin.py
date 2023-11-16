from nassl.ssl_client import ClientCertificateRequested
from sslyze.plugins.robot.implementation import RobotImplementation, RobotScanResult
from sslyze.plugins.robot._robot_tester import RobotScanResultEnum
from sslyze.server_setting import ServerNetworkLocation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import ClientAuthConfigEnum, LegacyOpenSslServer
import pytest

class TestRobotPluginPlugin:

    def test_robot_attack_good(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        server_location = ServerNetworkLocation('guide.duo.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result: RobotScanResult = RobotImplementation.scan_server(server_info)
        assert result.robot_result == RobotScanResultEnum.NOT_VULNERABLE_NO_ORACLE
        assert RobotImplementation.cli_connector_cls.result_to_console_output(result)

    @pytest.mark.skip('Not implemented; TODO: Find a vulnerable server.')
    def test_robot_attack_bad(self) -> None:
        if False:
            print('Hello World!')
        pass

    @can_only_run_on_linux_64
    def test_fails_when_client_auth_failed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            with pytest.raises(ClientCertificateRequested):
                RobotImplementation.scan_server(server_info)