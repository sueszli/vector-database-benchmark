import socket
import types
from dataclasses import dataclass
from typing import List, Optional
from nassl._nassl import WantReadError
from sslyze.json.pydantic_utils import BaseModelWithOrmModeAndForbid
from sslyze.json.scan_attempt_json import ScanCommandAttemptAsJson
from sslyze.plugins.plugin_base import ScanCommandResult, ScanCommandImplementation, ScanCommandExtraArgument, ScanJob, ScanCommandWrongUsageError, ScanCommandCliConnector, ScanJobResult
from tls_parser.alert_protocol import TlsAlertRecord
from tls_parser.application_data_protocol import TlsApplicationDataRecord
from tls_parser.change_cipher_spec_protocol import TlsChangeCipherSpecRecord
from tls_parser.exceptions import NotEnoughData, UnknownTlsVersionByte
from tls_parser.handshake_protocol import TlsHandshakeRecord, TlsHandshakeTypeByte
from tls_parser.parser import TlsRecordParser
import tls_parser.tls_version
from sslyze.server_connectivity import ServerConnectivityInfo, TlsVersionEnum

@dataclass(frozen=True)
class OpenSslCcsInjectionScanResult(ScanCommandResult):
    """The result of testing a server for the OpenSSL CCS Injection vulnerability (CVE-2014-0224).

    Attributes:
        is_vulnerable_to_ccs_injection: True if the server is vulnerable to the OpenSSL CCS Injection vulnerability.
    """
    is_vulnerable_to_ccs_injection: bool

class OpenSslCcsInjectionScanResultAsJson(BaseModelWithOrmModeAndForbid):
    is_vulnerable_to_ccs_injection: bool

class OpenSslCcsInjectionScanAttemptAsJson(ScanCommandAttemptAsJson):
    result: Optional[OpenSslCcsInjectionScanResultAsJson]

class _OpenSslCcsInjectionCliConnector(ScanCommandCliConnector[OpenSslCcsInjectionScanResult, None]):
    _cli_option = 'openssl_ccs'
    _cli_description = 'Test a server for the OpenSSL CCS Injection vulnerability (CVE-2014-0224).'

    @classmethod
    def result_to_console_output(cls, result: OpenSslCcsInjectionScanResult) -> List[str]:
        if False:
            return 10
        result_txt = [cls._format_title('OpenSSL CCS Injection')]
        ccs_text = 'VULNERABLE - Server is vulnerable to OpenSSL CCS injection' if result.is_vulnerable_to_ccs_injection else 'OK - Not vulnerable to OpenSSL CCS injection'
        result_txt.append(cls._format_field('', ccs_text))
        return result_txt

class OpenSslCcsInjectionImplementation(ScanCommandImplementation[OpenSslCcsInjectionScanResult, None]):
    """Test a server for the OpenSSL CCS Injection vulnerability (CVE-2014-0224)."""
    cli_connector_cls = _OpenSslCcsInjectionCliConnector

    @classmethod
    def scan_jobs_for_scan_command(cls, server_info: ServerConnectivityInfo, extra_arguments: Optional[ScanCommandExtraArgument]=None) -> List[ScanJob]:
        if False:
            while True:
                i = 10
        if extra_arguments:
            raise ScanCommandWrongUsageError('This plugin does not take extra arguments')
        return [ScanJob(function_to_call=_test_for_ccs_injection, function_arguments=[server_info])]

    @classmethod
    def result_for_completed_scan_jobs(cls, server_info: ServerConnectivityInfo, scan_job_results: List[ScanJobResult]) -> OpenSslCcsInjectionScanResult:
        if False:
            return 10
        if len(scan_job_results) != 1:
            raise RuntimeError(f'Unexpected number of scan jobs received: {scan_job_results}')
        return OpenSslCcsInjectionScanResult(is_vulnerable_to_ccs_injection=scan_job_results[0].get_result())

def _test_for_ccs_injection(server_info: ServerConnectivityInfo) -> bool:
    if False:
        i = 10
        return i + 15
    if server_info.tls_probing_result.highest_tls_version_supported.value >= TlsVersionEnum.TLS_1_3.value:
        return False
    ssl_connection = server_info.get_preconfigured_tls_connection(should_enable_server_name_indication=False)
    ssl_connection.ssl_client.do_handshake = types.MethodType(_do_handshake_with_ccs_injection, ssl_connection.ssl_client)
    is_vulnerable = False
    try:
        ssl_connection.connect()
    except _VulnerableToCcsInjection:
        is_vulnerable = True
    except _NotVulnerableToCcsInjection:
        pass
    finally:
        ssl_connection.close()
    return is_vulnerable

class _VulnerableToCcsInjection(Exception):
    """Exception to raise during the handshake to hijack the flow and test for CCS."""

class _NotVulnerableToCcsInjection(Exception):
    """Exception to raise during the handshake to hijack the flow and test for CCS."""

def _do_handshake_with_ccs_injection(self):
    if False:
        print('Hello World!')
    'Modified do_handshake() to send a CCS injection payload and return the result.'
    try:
        self._ssl.do_handshake()
    except WantReadError:
        len_to_read = self._network_bio.pending()
        while len_to_read:
            handshake_data_out = self._network_bio.read(len_to_read)
            self._sock.send(handshake_data_out)
            len_to_read = self._network_bio.pending()
    did_receive_hello_done = False
    remaining_bytes = b''
    while not did_receive_hello_done:
        try:
            (tls_record, len_consumed) = TlsRecordParser.parse_bytes(remaining_bytes)
            remaining_bytes = remaining_bytes[len_consumed:]
        except UnknownTlsVersionByte as e:
            if e.record_type == tls_parser.record_protocol.TlsRecordTypeByte.ALERT:
                raise _NotVulnerableToCcsInjection()
            else:
                raise
        except NotEnoughData:
            try:
                raw_ssl_bytes = self._sock.recv(16381)
            except ConnectionError:
                break
            if not raw_ssl_bytes:
                break
            remaining_bytes = remaining_bytes + raw_ssl_bytes
            continue
        if isinstance(tls_record, TlsHandshakeRecord):
            for handshake_message in tls_record.subprotocol_messages:
                if handshake_message.handshake_type == TlsHandshakeTypeByte.SERVER_DONE:
                    did_receive_hello_done = True
                    break
        elif isinstance(tls_record, TlsAlertRecord):
            break
        else:
            raise ValueError('Unknown record? Type {}'.format(tls_record.header.type))
    if did_receive_hello_done:
        payload = TlsChangeCipherSpecRecord.from_parameters(tls_version=tls_parser.tls_version.TlsVersionEnum[self._ssl_version.name]).to_bytes()
        self._sock.send(payload)
        app_data_record = TlsApplicationDataRecord.from_parameters(tls_version=tls_parser.tls_version.TlsVersionEnum[self._ssl_version.name], application_data=b'\x00\x00')
        self._sock.send(app_data_record.to_bytes())
        while True:
            try:
                (tls_record, len_consumed) = TlsRecordParser.parse_bytes(remaining_bytes)
                remaining_bytes = remaining_bytes[len_consumed:]
            except NotEnoughData:
                try:
                    raw_ssl_bytes = self._sock.recv(16381)
                    if not raw_ssl_bytes:
                        raise _NotVulnerableToCcsInjection()
                except socket.error:
                    raise _NotVulnerableToCcsInjection()
                remaining_bytes = remaining_bytes + raw_ssl_bytes
                continue
            if isinstance(tls_record, TlsAlertRecord):
                if tls_record.alert_description == 20:
                    raise _VulnerableToCcsInjection()
                raise _NotVulnerableToCcsInjection()
            else:
                break
        raise _NotVulnerableToCcsInjection()