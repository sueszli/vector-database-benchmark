from pathlib import Path
from cryptography.x509.ocsp import OCSPResponseStatus
from sslyze.plugins.certificate_info.implementation import CertificateInfoImplementation, CertificateInfoExtraArgument
from sslyze.server_setting import ServerNetworkLocation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import ModernOpenSslServer, ClientAuthConfigEnum
import pytest

class TestCertificateInfoPlugin:

    def test_ca_file_bad_file(self):
        if False:
            return 10
        server_location = ServerNetworkLocation('www.hotmail.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        with pytest.raises(ValueError):
            CertificateInfoImplementation.scan_server(server_info, CertificateInfoExtraArgument(custom_ca_file=Path('doesntexist')))

    def test_ca_file(self):
        if False:
            for i in range(10):
                print('nop')
        server_location = ServerNetworkLocation('www.hotmail.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        ca_file_path = Path(__file__).parent / '..' / '..' / 'certificates' / 'wildcard-self-signed.pem'
        plugin_result = CertificateInfoImplementation.scan_server(server_info, CertificateInfoExtraArgument(custom_ca_file=ca_file_path))
        assert len(plugin_result.certificate_deployments[0].path_validation_results) >= 6
        for path_validation_result in plugin_result.certificate_deployments[0].path_validation_results:
            if path_validation_result.trust_store.path == ca_file_path:
                assert not path_validation_result.was_validation_successful
            else:
                assert path_validation_result.was_validation_successful

    def test_valid_chain_with_ocsp_stapling(self):
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation('www.microsoft.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert plugin_result.certificate_deployments[0].ocsp_response
        assert plugin_result.certificate_deployments[0].ocsp_response.response_status == OCSPResponseStatus.SUCCESSFUL
        assert plugin_result.certificate_deployments[0].ocsp_response_is_trusted
        assert not plugin_result.certificate_deployments[0].leaf_certificate_has_must_staple_extension

    def test_valid_chain_with_ev_cert(self):
        if False:
            while True:
                i = 10
        server_location = ServerNetworkLocation('www.digicert.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert plugin_result.certificate_deployments[0].leaf_certificate_is_ev
        assert len(plugin_result.certificate_deployments[0].received_certificate_chain)
        assert len(plugin_result.certificate_deployments[0].verified_certificate_chain)
        assert not plugin_result.certificate_deployments[0].received_chain_contains_anchor_certificate
        assert len(plugin_result.certificate_deployments[0].path_validation_results) == 5
        for path_validation_result in plugin_result.certificate_deployments[0].path_validation_results:
            assert path_validation_result.was_validation_successful
        assert plugin_result.certificate_deployments[0].leaf_certificate_subject_matches_hostname
        assert plugin_result.certificate_deployments[0].received_chain_has_valid_order

    def test_invalid_chain(self):
        if False:
            while True:
                i = 10
        server_location = ServerNetworkLocation('self-signed.badssl.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert not plugin_result.certificate_deployments[0].verified_certificate_chain
        assert plugin_result.certificate_deployments[0].verified_chain_has_sha1_signature is None
        assert plugin_result.certificate_deployments[0].ocsp_response is None
        assert len(plugin_result.certificate_deployments[0].received_certificate_chain) == 1
        assert len(plugin_result.certificate_deployments[0].path_validation_results) >= 5
        for path_validation_result in plugin_result.certificate_deployments[0].path_validation_results:
            assert not path_validation_result.was_validation_successful
        assert plugin_result.certificate_deployments[0].leaf_certificate_signed_certificate_timestamps_count == 0
        assert plugin_result.certificate_deployments[0].leaf_certificate_subject_matches_hostname
        assert plugin_result.certificate_deployments[0].received_chain_has_valid_order
        assert plugin_result.certificate_deployments[0].received_chain_contains_anchor_certificate is None

    def test_1000_sans_chain(self):
        if False:
            for i in range(10):
                print('nop')
        server_location = ServerNetworkLocation('1000-sans.badssl.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        CertificateInfoImplementation.scan_server(server_info)

    @pytest.mark.skip('Can no longer build a verified because CA cert expired')
    def test_sha1_chain(self):
        if False:
            while True:
                i = 10
        server_location = ServerNetworkLocation('sha1-intermediate.badssl.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert plugin_result.certificate_deployments[0].verified_chain_has_sha1_signature

    def test_sha256_chain(self):
        if False:
            for i in range(10):
                print('nop')
        server_location = ServerNetworkLocation('sha256.badssl.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert not plugin_result.certificate_deployments[0].verified_chain_has_sha1_signature

    def test_chain_with_anchor(self):
        if False:
            return 10
        server_location = ServerNetworkLocation('www.verizon.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert plugin_result.certificate_deployments[0].received_chain_contains_anchor_certificate

    def test_certificate_with_no_cn(self):
        if False:
            return 10
        server_location = ServerNetworkLocation('no-common-name.badssl.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert plugin_result.certificate_deployments[0].received_certificate_chain

    def test_certificate_with_no_subject(self):
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation('no-subject.badssl.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert plugin_result.certificate_deployments[0].received_certificate_chain

    def test_certificate_with_scts(self):
        if False:
            return 10
        server_location = ServerNetworkLocation('www.apple.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert plugin_result.certificate_deployments[0].leaf_certificate_signed_certificate_timestamps_count > 1

    def test_multiple_certificates(self):
        if False:
            return 10
        server_location = ServerNetworkLocation('www.facebook.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        plugin_result = CertificateInfoImplementation.scan_server(server_info)
        assert len(plugin_result.certificate_deployments) > 1

    @can_only_run_on_linux_64
    def test_succeeds_when_client_auth_failed(self):
        if False:
            print('Hello World!')
        with ModernOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, port=server.port, ip_address=server.ip_address)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            plugin_result = CertificateInfoImplementation.scan_server(server_info)
            assert plugin_result.certificate_deployments[0].received_certificate_chain