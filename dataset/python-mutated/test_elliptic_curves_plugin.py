from sslyze import ServerNetworkLocation
from sslyze.plugins.elliptic_curves_plugin import SupportedEllipticCurvesScanResult, SupportedEllipticCurvesImplementation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import ModernOpenSslServer

class TestEllipticCurvesPluginWithOnlineServer:

    def test_supported_curves(self) -> None:
        if False:
            return 10
        server_location = ServerNetworkLocation('www.cloudflare.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result: SupportedEllipticCurvesScanResult = SupportedEllipticCurvesImplementation.scan_server(server_info)
        assert result.supports_ecdh_key_exchange
        assert result.supported_curves
        assert result.rejected_curves
        assert SupportedEllipticCurvesImplementation.cli_connector_cls.result_to_console_output(result)

@can_only_run_on_linux_64
class TestEllipticCurvesPluginWithLocalServer:

    def test_supported_curves(self) -> None:
        if False:
            return 10
        server_curves = ['X25519', 'X448', 'prime256v1', 'secp384r1', 'secp521r1']
        with ModernOpenSslServer(groups=':'.join(server_curves)) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result: SupportedEllipticCurvesScanResult = SupportedEllipticCurvesImplementation.scan_server(server_info)
        assert result.supported_curves
        assert set(server_curves) == {curve.name for curve in result.supported_curves}