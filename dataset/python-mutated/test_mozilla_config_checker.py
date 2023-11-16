import pytest
from sslyze import Scanner, ServerScanRequest, ServerNetworkLocation
from sslyze.mozilla_tls_profile.mozilla_config_checker import MozillaTlsConfigurationChecker, MozillaTlsConfigurationEnum, ServerNotCompliantWithMozillaTlsConfiguration, ServerScanResultIncomplete
from tests.factories import ServerScanResultFactory

@pytest.fixture(scope='session')
def server_scan_result_for_google():
    if False:
        i = 10
        return i + 15
    scanner = Scanner()
    scanner.queue_scans([ServerScanRequest(server_location=ServerNetworkLocation(hostname='google.com'))])
    for server_scan_result in scanner.get_results():
        yield server_scan_result

class TestMozillaTlsConfigurationChecker:

    @pytest.mark.skip('Server needs to be updated; check https://github.com/chromium/badssl.com/issues/483')
    def test_badssl_compliant_with_old(self):
        if False:
            print('Hello World!')
        scanner = Scanner()
        scanner.queue_scans([ServerScanRequest(server_location=ServerNetworkLocation(hostname='mozilla-old.badssl.com'))])
        server_scan_result = next(scanner.get_results())
        checker = MozillaTlsConfigurationChecker.get_default()
        checker.check_server(against_config=MozillaTlsConfigurationEnum.OLD, server_scan_result=server_scan_result)
        for mozilla_config in [MozillaTlsConfigurationEnum.INTERMEDIATE, MozillaTlsConfigurationEnum.MODERN]:
            with pytest.raises(ServerNotCompliantWithMozillaTlsConfiguration):
                checker.check_server(against_config=mozilla_config, server_scan_result=server_scan_result)

    @pytest.mark.skip('Server needs to be updated; check https://github.com/chromium/badssl.com/issues/483')
    def test_badssl_compliant_with_intermediate(self):
        if False:
            while True:
                i = 10
        scanner = Scanner()
        scanner.queue_scans([ServerScanRequest(server_location=ServerNetworkLocation(hostname='mozilla-intermediate.badssl.com'))])
        server_scan_result = next(scanner.get_results())
        checker = MozillaTlsConfigurationChecker.get_default()
        checker.check_server(against_config=MozillaTlsConfigurationEnum.INTERMEDIATE, server_scan_result=server_scan_result)
        for mozilla_config in [MozillaTlsConfigurationEnum.OLD, MozillaTlsConfigurationEnum.MODERN]:
            with pytest.raises(ServerNotCompliantWithMozillaTlsConfiguration):
                checker.check_server(against_config=mozilla_config, server_scan_result=server_scan_result)

    @pytest.mark.skip('Server needs to be updated; check https://github.com/chromium/badssl.com/issues/483')
    def test_badssl_compliant_with_modern(self):
        if False:
            print('Hello World!')
        scanner = Scanner()
        scanner.queue_scans([ServerScanRequest(server_location=ServerNetworkLocation(hostname='mozilla-modern.badssl.com'))])
        server_scan_result = next(scanner.get_results())
        checker = MozillaTlsConfigurationChecker.get_default()
        checker.check_server(against_config=MozillaTlsConfigurationEnum.MODERN, server_scan_result=server_scan_result)
        for mozilla_config in [MozillaTlsConfigurationEnum.OLD, MozillaTlsConfigurationEnum.INTERMEDIATE]:
            with pytest.raises(ServerNotCompliantWithMozillaTlsConfiguration):
                checker.check_server(against_config=mozilla_config, server_scan_result=server_scan_result)

    def test_multi_certs_deployment_compliant_with_old(self, server_scan_result_for_google):
        if False:
            for i in range(10):
                print('nop')
        checker = MozillaTlsConfigurationChecker.get_default()
        checker.check_server(against_config=MozillaTlsConfigurationEnum.OLD, server_scan_result=server_scan_result_for_google)

    def test_multi_certs_deployment_not_compliant_with_intermediate(self, server_scan_result_for_google):
        if False:
            while True:
                i = 10
        checker = MozillaTlsConfigurationChecker.get_default()
        with pytest.raises(ServerNotCompliantWithMozillaTlsConfiguration):
            checker.check_server(against_config=MozillaTlsConfigurationEnum.INTERMEDIATE, server_scan_result=server_scan_result_for_google)

    def test_multi_certs_deployment_not_compliant_with_modern(self, server_scan_result_for_google):
        if False:
            print('Hello World!')
        checker = MozillaTlsConfigurationChecker.get_default()
        with pytest.raises(ServerNotCompliantWithMozillaTlsConfiguration):
            checker.check_server(against_config=MozillaTlsConfigurationEnum.MODERN, server_scan_result=server_scan_result_for_google)

    def test_incomplete_scan_result(self):
        if False:
            for i in range(10):
                print('nop')
        server_scan_result = ServerScanResultFactory.create()
        checker = MozillaTlsConfigurationChecker.get_default()
        with pytest.raises(ServerScanResultIncomplete):
            checker.check_server(against_config=MozillaTlsConfigurationEnum.MODERN, server_scan_result=server_scan_result)