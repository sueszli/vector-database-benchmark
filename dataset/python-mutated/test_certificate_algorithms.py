from pathlib import Path
from unittest import mock
from unittest.mock import PropertyMock
import cryptography
import pytest
from sslyze.plugins.certificate_info.json_output import CertificateInfoScanResultAsJson
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import ModernOpenSslServer
from sslyze import ServerNetworkLocation
from sslyze.plugins.certificate_info.implementation import CertificateInfoImplementation

class TestCertificateAlgorithms:

    @can_only_run_on_linux_64
    def test_rsa_certificate(self):
        if False:
            return 10
        with ModernOpenSslServer() as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, port=server.port, ip_address=server.ip_address)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            scan_result = CertificateInfoImplementation.scan_server(server_info)
            assert scan_result.certificate_deployments[0].received_certificate_chain
            result_as_json = CertificateInfoScanResultAsJson.from_orm(scan_result).json()
            assert result_as_json
            result_as_txt = CertificateInfoImplementation.cli_connector_cls.result_to_console_output(scan_result)
            assert result_as_txt

    @can_only_run_on_linux_64
    def test_ed25519_certificate(self):
        if False:
            return 10
        with ModernOpenSslServer(server_certificate_path=Path(__file__).parent.absolute() / 'server-ed25519-cert.pem', server_key_path=Path(__file__).parent.absolute() / 'server-ed25519-key.pem') as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, port=server.port, ip_address=server.ip_address)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            scan_result = CertificateInfoImplementation.scan_server(server_info)
            assert scan_result.certificate_deployments[0].received_certificate_chain
            result_as_json = CertificateInfoScanResultAsJson.from_orm(scan_result).json()
            assert result_as_json
            result_as_txt = CertificateInfoImplementation.cli_connector_cls.result_to_console_output(scan_result)
            assert result_as_txt

    def test_ecdsa_certificate(self):
        if False:
            while True:
                i = 10
        server_location = ServerNetworkLocation('www.cloudflare.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        scan_result = CertificateInfoImplementation.scan_server(server_info)
        result_as_json = CertificateInfoScanResultAsJson.from_orm(scan_result).json()
        assert result_as_json
        result_as_txt = CertificateInfoImplementation.cli_connector_cls.result_to_console_output(scan_result)
        assert result_as_txt

    @pytest.mark.parametrize('certificate_name_field', ['subject', 'issuer'])
    def test_invalid_certificate_bad_name(self, certificate_name_field):
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation('www.cloudflare.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        with mock.patch.object(cryptography.x509.Certificate, certificate_name_field, new_callable=PropertyMock) as mock_certificate_name:
            mock_certificate_name.side_effect = ValueError('Country name must be a 2 character country code')
            scan_result = CertificateInfoImplementation.scan_server(server_info)
            result_as_txt = CertificateInfoImplementation.cli_connector_cls.result_to_console_output(scan_result)
            assert result_as_txt
            result_as_json = CertificateInfoScanResultAsJson.from_orm(scan_result).json()
            assert result_as_json