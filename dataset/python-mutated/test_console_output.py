from io import StringIO
from sslyze.cli.console_output import ObserverToGenerateConsoleOutput
from sslyze.plugins.compression_plugin import CompressionScanResult
from sslyze import ScanCommandErrorReasonEnum, ScanCommandAttemptStatusEnum
from sslyze.scanner.models import CompressionScanAttempt
from sslyze.server_connectivity import ClientAuthRequirementEnum
from tests.factories import ServerScanResultFactory, TracebackExceptionFactory, ServerNetworkLocationViaHttpProxyFactory, ParsedCommandLineFactory, ConnectionToServerFailedFactory, ServerScanRequestFactory, ServerTlsProbingResultFactory, AllScanCommandsAttemptsFactory

class TestObserverToGenerateConsoleOutput:

    def test_command_line_parsed(self):
        if False:
            print('Hello World!')
        parsed_cmd_line = ParsedCommandLineFactory.create()
        assert parsed_cmd_line.invalid_servers
        assert parsed_cmd_line.servers_to_scans
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.command_line_parsed(parsed_cmd_line)
            final_output = file_out.getvalue()
        assert final_output
        for bad_server in parsed_cmd_line.invalid_servers:
            assert bad_server.server_string in final_output
            assert bad_server.error_message in final_output

    def test_server_connectivity_test_error(self):
        if False:
            return 10
        scan_request = ServerScanRequestFactory.create()
        error = ConnectionToServerFailedFactory.create()
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.server_connectivity_test_error(scan_request, error)
            final_output = file_out.getvalue()
        assert final_output
        assert error.error_message in final_output

    def test_server_connectivity_test_completed(self):
        if False:
            while True:
                i = 10
        scan_request = ServerScanRequestFactory.create()
        connectivity_result = ServerTlsProbingResultFactory.create()
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.server_connectivity_test_completed(scan_request, connectivity_result)
            final_output = file_out.getvalue()
        assert final_output
        assert scan_request.server_location.hostname in final_output

    def test_server_connectivity_test_completed_with_required_client_auth(self):
        if False:
            return 10
        scan_request = ServerScanRequestFactory.create()
        connectivity_result = ServerTlsProbingResultFactory.create(client_auth_requirement=ClientAuthRequirementEnum.REQUIRED)
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.server_connectivity_test_completed(scan_request, connectivity_result)
            final_output = file_out.getvalue()
        assert final_output
        assert 'Server REQUIRED client authentication' in final_output

    def test_server_connectivity_test_completed_with_http_tunneling(self):
        if False:
            return 10
        scan_request = ServerScanRequestFactory.create(server_location=ServerNetworkLocationViaHttpProxyFactory.create())
        connectivity_result = ServerTlsProbingResultFactory.create()
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.server_connectivity_test_completed(scan_request, connectivity_result)
            final_output = file_out.getvalue()
        assert final_output
        assert 'proxy' in final_output

    def test_server_scan_completed(self):
        if False:
            print('Hello World!')
        compression_attempt = CompressionScanAttempt(status=ScanCommandAttemptStatusEnum.COMPLETED, error_reason=None, error_trace=None, result=CompressionScanResult(supports_compression=True))
        scan_result = ServerScanResultFactory.create(scan_result=AllScanCommandsAttemptsFactory.create({'tls_compression': compression_attempt}))
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.server_scan_completed(scan_result)
            final_output = file_out.getvalue()
        assert final_output
        assert 'Compression' in final_output

    def test_server_scan_completed_with_proxy(self):
        if False:
            return 10
        compression_attempt = CompressionScanAttempt(status=ScanCommandAttemptStatusEnum.COMPLETED, error_reason=None, error_trace=None, result=CompressionScanResult(supports_compression=True))
        scan_result = ServerScanResultFactory.create(server_location=ServerNetworkLocationViaHttpProxyFactory.create(), scan_result=AllScanCommandsAttemptsFactory.create({'tls_compression': compression_attempt}))
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.server_scan_completed(scan_result)
            final_output = file_out.getvalue()
        assert final_output
        assert 'HTTP PROXY' in final_output
        assert 'Compression' in final_output

    def test_server_scan_completed_with_error(self):
        if False:
            i = 10
            return i + 15
        error_trace = TracebackExceptionFactory.create()
        compression_attempt = CompressionScanAttempt(status=ScanCommandAttemptStatusEnum.ERROR, error_reason=ScanCommandErrorReasonEnum.BUG_IN_SSLYZE, error_trace=error_trace, result=None)
        scan_result = ServerScanResultFactory.create(scan_result=AllScanCommandsAttemptsFactory.create({'tls_compression': compression_attempt}))
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.server_scan_completed(scan_result)
            final_output = file_out.getvalue()
        assert final_output
        assert error_trace.stack.format()[0] in final_output

    def test_scans_completed(self):
        if False:
            for i in range(10):
                print('nop')
        with StringIO() as file_out:
            console_gen = ObserverToGenerateConsoleOutput(file_to=file_out)
            console_gen.all_server_scans_completed()