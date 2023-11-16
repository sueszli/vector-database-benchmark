from sslyze.plugins.certificate_info.implementation import CertificateInfoImplementation
from sslyze.plugins.certificate_info.json_output import CertificateInfoScanResultAsJson
from sslyze.server_setting import ServerNetworkLocation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info

class TestJsonEncoder:

    def test(self):
        if False:
            return 10
        server_location = ServerNetworkLocation('www.facebook.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        result_as_json = CertificateInfoScanResultAsJson.from_orm(plugin_result).json()
        assert result_as_json
        assert 'issuer' in result_as_json
        assert 'subject' in result_as_json