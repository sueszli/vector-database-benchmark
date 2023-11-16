"""Script designed to scan a server running on localhost:443.

This is used in CI to ensure SSLyze is able to scan specific web servers (Apache2, IIS, etc.) running on localhost:443
with client authentication required.
.
See ./.github/workflows and https://github.com/nabla-c0d3/sslyze/issues/472 for more details.

$ PYTHONPATH=. python tests/web_servers/scan_localhost.py apache2
"""
import sys
from datetime import datetime
from enum import Enum
from sslyze import Scanner, ServerScanRequest, ClientAuthRequirementEnum, ScanCommandErrorReasonEnum, ServerNetworkLocation, ServerConnectivityStatusEnum, ScanCommandAttemptStatusEnum
from sslyze.json.json_output import SslyzeOutputAsJson, ServerScanResultAsJson
from sslyze.plugins.scan_commands import ScanCommandsRepository, ScanCommand

class WebServerSoftwareEnum(str, Enum):
    APACHE2 = 'apache2'
    NGINX = 'nginx'
    IIS = 'iis'

def main(server_software_running_on_localhost: WebServerSoftwareEnum) -> None:
    if False:
        for i in range(10):
            print('nop')
    print('Starting scan.')
    date_scans_started = datetime.utcnow()
    scanner = Scanner()
    scanner.queue_scans([ServerScanRequest(server_location=ServerNetworkLocation('localhost', 443))])
    for server_scan_result in scanner.get_results():
        assert server_scan_result.connectivity_status == ServerConnectivityStatusEnum.COMPLETED
        assert server_scan_result.connectivity_result
        if server_software_running_on_localhost == WebServerSoftwareEnum.APACHE2:
            if server_scan_result.connectivity_result.client_auth_requirement != ClientAuthRequirementEnum.REQUIRED:
                raise RuntimeError(f'SSLyze did not detect that client authentication was required by Apache2: {server_scan_result.connectivity_result.client_auth_requirement}.')
        elif server_software_running_on_localhost == WebServerSoftwareEnum.NGINX:
            if server_scan_result.connectivity_result.client_auth_requirement != ClientAuthRequirementEnum.OPTIONAL:
                raise RuntimeError(f'SSLyze did not detect that client authentication was required by Nginx: {server_scan_result.connectivity_result.client_auth_requirement}.')
        elif server_software_running_on_localhost == WebServerSoftwareEnum.IIS:
            if server_scan_result.connectivity_result.client_auth_requirement != ClientAuthRequirementEnum.DISABLED:
                raise RuntimeError(f'SSLyze detected that client authentication was enabled by IIS: {server_scan_result.connectivity_result.client_auth_requirement}.')
        else:
            raise ValueError(f'Unexpected value: {server_software_running_on_localhost}')
        successful_cmds = set()
        triggered_unexpected_error = False
        for scan_command in ScanCommandsRepository.get_all_scan_commands():
            scan_cmd_attempt = getattr(server_scan_result.scan_result, scan_command.value)
            if scan_cmd_attempt.status == ScanCommandAttemptStatusEnum.COMPLETED:
                successful_cmds.add(scan_command)
            elif scan_cmd_attempt.status == ScanCommandAttemptStatusEnum.ERROR:
                if scan_cmd_attempt.error_reason != ScanCommandErrorReasonEnum.CLIENT_CERTIFICATE_NEEDED:
                    triggered_unexpected_error = True
                    print(f'\nError when running {scan_command}: {scan_cmd_attempt.error_reason}.')
                    if scan_cmd_attempt.error_trace:
                        exc_trace = ''
                        for line in scan_cmd_attempt.error_trace.format(chain=False):
                            exc_trace += f'       {line}'
                        print(exc_trace)
        print(f'Finished scan with {len(successful_cmds)} results.')
        if triggered_unexpected_error:
            raise RuntimeError('The scan triggered unexpected errors')
        else:
            print('OK: Triggered CLIENT_CERTIFICATE_NEEDED errors only.')
        if server_software_running_on_localhost == WebServerSoftwareEnum.APACHE2:
            expected_scan_cmds_to_succeed = {ScanCommand.TLS_1_3_CIPHER_SUITES, ScanCommand.TLS_1_2_CIPHER_SUITES, ScanCommand.TLS_1_1_CIPHER_SUITES, ScanCommand.TLS_1_0_CIPHER_SUITES, ScanCommand.SSL_3_0_CIPHER_SUITES, ScanCommand.SSL_2_0_CIPHER_SUITES, ScanCommand.OPENSSL_CCS_INJECTION, ScanCommand.HEARTBLEED, ScanCommand.ELLIPTIC_CURVES, ScanCommand.TLS_FALLBACK_SCSV, ScanCommand.CERTIFICATE_INFO, ScanCommand.TLS_COMPRESSION}
        elif server_software_running_on_localhost == WebServerSoftwareEnum.NGINX:
            expected_scan_cmds_to_succeed = {ScanCommand.TLS_1_3_CIPHER_SUITES, ScanCommand.TLS_1_2_CIPHER_SUITES, ScanCommand.TLS_1_1_CIPHER_SUITES, ScanCommand.TLS_1_0_CIPHER_SUITES, ScanCommand.SSL_3_0_CIPHER_SUITES, ScanCommand.SSL_2_0_CIPHER_SUITES, ScanCommand.OPENSSL_CCS_INJECTION, ScanCommand.HEARTBLEED, ScanCommand.ELLIPTIC_CURVES, ScanCommand.TLS_FALLBACK_SCSV, ScanCommand.CERTIFICATE_INFO, ScanCommand.TLS_COMPRESSION, ScanCommand.SESSION_RESUMPTION, ScanCommand.TLS_1_3_EARLY_DATA, ScanCommand.HTTP_HEADERS, ScanCommand.SESSION_RENEGOTIATION}
        elif server_software_running_on_localhost == WebServerSoftwareEnum.IIS:
            expected_scan_cmds_to_succeed = ScanCommandsRepository.get_all_scan_commands()
        else:
            raise ValueError(f'Unexpected value: {server_software_running_on_localhost}')
        missing_scan_cmds = expected_scan_cmds_to_succeed.difference(successful_cmds)
        if missing_scan_cmds:
            raise RuntimeError(f'SSLyze did not complete all the expected scan commands: {missing_scan_cmds}')
        print('OK: Completed all the expected scan commands.')
        if server_software_running_on_localhost in [WebServerSoftwareEnum.APACHE2, WebServerSoftwareEnum.NGINX]:
            expected_enabled_tls_scan_commands = {ScanCommand.TLS_1_3_CIPHER_SUITES, ScanCommand.TLS_1_2_CIPHER_SUITES}
        elif server_software_running_on_localhost == WebServerSoftwareEnum.IIS:
            expected_enabled_tls_scan_commands = {ScanCommand.TLS_1_2_CIPHER_SUITES, ScanCommand.TLS_1_1_CIPHER_SUITES, ScanCommand.TLS_1_0_CIPHER_SUITES}
        else:
            raise ValueError(f'Unexpected value: {server_software_running_on_localhost}')
        for ciphers_scan_cmd in expected_enabled_tls_scan_commands:
            scan_cmd_attempt = getattr(server_scan_result.scan_result, ciphers_scan_cmd, None)
            scan_cmd_result = scan_cmd_attempt.result
            if not scan_cmd_result.accepted_cipher_suites:
                raise RuntimeError(f'SSLyze did not detect {scan_cmd_result.tls_version_used.name} to be enabled on the server.')
            else:
                print(f'OK: Scan command {ciphers_scan_cmd} detected cipher suites.')
        final_json_output = SslyzeOutputAsJson(server_scan_results=[ServerScanResultAsJson.from_orm(server_scan_result)], date_scans_started=date_scans_started, date_scans_completed=datetime.utcnow())
        final_json_output.json()
        print('OK: Was able to generate JSON output.')
if __name__ == '__main__':
    server_argument = WebServerSoftwareEnum(sys.argv[1])
    main(server_argument)