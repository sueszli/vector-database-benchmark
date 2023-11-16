from unittest import mock
from sslyze import Scanner, ServerScanRequest, ServerScanStatusEnum, ServerScanResult, ServerTlsProbingResult, ServerNetworkLocation, ScanCommand, ScanCommandAttemptStatusEnum, ScanCommandErrorReasonEnum
from sslyze.errors import ConnectionToServerFailed
from sslyze.scanner import _mass_connectivity_tester
from sslyze.scanner.scanner_observer import ScannerObserver
from tests.factories import ServerScanRequestFactory, ServerTlsProbingResultFactory
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import LegacyOpenSslServer, ClientAuthConfigEnum

class _MockScannerObserver(ScannerObserver):

    def __init__(self):
        if False:
            return 10
        self.server_connectivity_test_error_calls_count = 0
        self.server_connectivity_test_completed_calls_count = 0
        self.server_scan_completed_calls_count = 0
        self.all_server_scans_completed_calls_count = 0

    def server_connectivity_test_error(self, server_scan_request: ServerScanRequest, connectivity_error: ConnectionToServerFailed) -> None:
        if False:
            return 10
        self.server_connectivity_test_error_calls_count += 1

    def server_connectivity_test_completed(self, server_scan_request: ServerScanRequest, connectivity_result: ServerTlsProbingResult) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.server_connectivity_test_completed_calls_count += 1

    def server_scan_completed(self, server_scan_result: ServerScanResult) -> None:
        if False:
            print('Hello World!')
        self.server_scan_completed_calls_count += 1

    def all_server_scans_completed(self) -> None:
        if False:
            while True:
                i = 10
        self.all_server_scans_completed_calls_count += 1

class TestScanner:

    def test(self, mock_scan_commands):
        if False:
            i = 10
            return i + 15
        all_scan_requests = [ServerScanRequestFactory.create() for _ in range(20)]
        connectivity_result = ServerTlsProbingResultFactory.create()
        with mock.patch.object(_mass_connectivity_tester, 'check_connectivity_to_server', return_value=connectivity_result):
            observer = _MockScannerObserver()
            scanner = Scanner(observers=[observer])
            scanner.queue_scans(all_scan_requests)
            assert scanner._has_started_work
            all_scan_results = []
            for result in scanner.get_results():
                all_scan_results.append(result)
        assert len(all_scan_results) == len(all_scan_requests)
        assert {result.scan_status for result in all_scan_results} == {ServerScanStatusEnum.COMPLETED}
        assert observer.server_connectivity_test_error_calls_count == 0
        assert observer.server_connectivity_test_completed_calls_count == len(all_scan_requests)
        assert observer.server_scan_completed_calls_count == len(all_scan_requests)
        assert observer.all_server_scans_completed_calls_count == 1

    def test_connectivity_error(self, mock_scan_commands):
        if False:
            while True:
                i = 10
        scan_request = ServerScanRequestFactory.create()
        error = ConnectionToServerFailed(server_location=scan_request.server_location, network_configuration=scan_request.network_configuration, error_message='testt')
        with mock.patch.object(_mass_connectivity_tester, 'check_connectivity_to_server', side_effect=error):
            observer = _MockScannerObserver()
            scanner = Scanner(observers=[observer])
            scanner.queue_scans([scan_request])
            all_scan_results = []
            for result in scanner.get_results():
                all_scan_results.append(result)
        assert len(all_scan_results) == 1
        assert all_scan_results[0].scan_status == ServerScanStatusEnum.ERROR_NO_CONNECTIVITY
        assert observer.server_connectivity_test_error_calls_count == 1
        assert observer.server_connectivity_test_completed_calls_count == 0
        assert observer.server_scan_completed_calls_count == 1
        assert observer.all_server_scans_completed_calls_count == 1

    @can_only_run_on_linux_64
    def test_error_client_certificate_needed(self):
        if False:
            i = 10
            return i + 15
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            scan_request = ServerScanRequest(server_location=ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port), scan_commands={ScanCommand.HTTP_HEADERS})
            scanner = Scanner()
            scanner.queue_scans([scan_request])
            all_results = []
            for result in scanner.get_results():
                all_results.append(result)
        assert len(all_results) == 1
        http_headers_result = all_results[0].scan_result.http_headers
        assert http_headers_result.status == ScanCommandAttemptStatusEnum.ERROR
        assert http_headers_result.error_reason == ScanCommandErrorReasonEnum.CLIENT_CERTIFICATE_NEEDED
        assert http_headers_result.error_trace
        assert http_headers_result.result is None