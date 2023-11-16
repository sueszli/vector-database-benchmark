from sslyze.plugins.certificate_info.implementation import CertificateInfoImplementation
from sslyze.server_setting import ServerNetworkLocation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info

class TestCertificateInfoCliConnector:

    def test(self):
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation('www.facebook.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        result_as_txt = CertificateInfoImplementation.cli_connector_cls.result_to_console_output(plugin_result)
        assert result_as_txt