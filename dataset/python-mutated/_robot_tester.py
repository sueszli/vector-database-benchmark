import socket
import types
from enum import Enum
from typing import Optional, List, Dict
import binascii
import math
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey, RSAPublicNumbers
from cryptography.x509 import load_pem_x509_certificate
from nassl._nassl import WantReadError
from nassl.ssl_client import ClientCertificateRequested
from tls_parser.change_cipher_spec_protocol import TlsChangeCipherSpecRecord
from tls_parser.alert_protocol import TlsAlertRecord
from tls_parser.record_protocol import TlsRecordTlsVersionBytes
from tls_parser.exceptions import NotEnoughData
from tls_parser.handshake_protocol import TlsHandshakeRecord, TlsHandshakeTypeByte, TlsRsaClientKeyExchangeRecord
from tls_parser.parser import TlsRecordParser
import tls_parser.tls_version
from sslyze.errors import ServerRejectedTlsHandshake
from sslyze.server_connectivity import ServerConnectivityInfo, TlsVersionEnum, ClientAuthRequirementEnum

class RobotScanResultEnum(str, Enum):
    """The result of attempting exploit the ROBOT issue on the server.

    Attributes:
        VULNERABLE_WEAK_ORACLE: The server is vulnerable but the attack would take too long.
        VULNERABLE_STRONG_ORACLE: The server is vulnerable and real attacks are feasible.
        NOT_VULNERABLE_NO_ORACLE: The server supports RSA cipher suites but does not act as an oracle.
        NOT_VULNERABLE_RSA_NOT_SUPPORTED: The server does not supports RSA cipher suites.
        UNKNOWN_INCONSISTENT_RESULTS: Could not determine whether the server is vulnerable or not.
    """
    VULNERABLE_WEAK_ORACLE = 'VULNERABLE_WEAK_ORACLE'
    VULNERABLE_STRONG_ORACLE = 'VULNERABLE_STRONG_ORACLE'
    NOT_VULNERABLE_NO_ORACLE = 'NOT_VULNERABLE_NO_ORACLE'
    NOT_VULNERABLE_RSA_NOT_SUPPORTED = 'NOT_VULNERABLE_RSA_NOT_SUPPORTED'
    UNKNOWN_INCONSISTENT_RESULTS = 'UNKNOWN_INCONSISTENT_RESULTS'

class RobotPmsPaddingPayloadEnum(Enum):
    VALID = 0
    WRONG_FIRST_TWO_BYTES = 1
    WRONG_POSITION_00 = 2
    NO_00_IN_THE_MIDDLE = 3
    WRONG_VERSION_NUMBER = 4

class _RobotTlsRecordPayloads:
    _CKE_PAYLOADS_HEX = {RobotPmsPaddingPayloadEnum.VALID: '0002{pms_padding}00{tls_version}{pms}', RobotPmsPaddingPayloadEnum.WRONG_FIRST_TWO_BYTES: '4117{pms_padding}00{tls_version}{pms}', RobotPmsPaddingPayloadEnum.WRONG_POSITION_00: '0002{pms_padding}11{pms}0011', RobotPmsPaddingPayloadEnum.NO_00_IN_THE_MIDDLE: '0002{pms_padding}111111{pms}', RobotPmsPaddingPayloadEnum.WRONG_VERSION_NUMBER: '0002{pms_padding}000202{pms}'}
    _PMS_HEX = 'aa112233445566778899112233445566778899112233445566778899112233445566778899112233445566778899'

    @classmethod
    def get_client_key_exchange_record(cls, robot_payload_enum: RobotPmsPaddingPayloadEnum, tls_version: tls_parser.tls_version.TlsVersionEnum, modulus: int, exponent: int) -> TlsRsaClientKeyExchangeRecord:
        if False:
            while True:
                i = 10
        'A client key exchange record with a hardcoded pre_master_secret, and a valid or invalid padding.'
        pms_padding = cls._compute_pms_padding(modulus)
        tls_version_hex = binascii.b2a_hex(TlsRecordTlsVersionBytes[tls_version.name].value).decode('ascii')
        pms_with_padding_payload = cls._CKE_PAYLOADS_HEX[robot_payload_enum]
        final_pms = pms_with_padding_payload.format(pms_padding=pms_padding, tls_version=tls_version_hex, pms=cls._PMS_HEX)
        cke_robot_record = TlsRsaClientKeyExchangeRecord.from_parameters(tls_version, exponent, modulus, int(final_pms, 16))
        return cke_robot_record

    @staticmethod
    def _compute_pms_padding(modulus: int) -> str:
        if False:
            print('Hello World!')
        modulus_bit_size = int(math.ceil(math.log(modulus, 2)))
        modulus_byte_size = (modulus_bit_size + 7) // 8
        pad_len = (modulus_byte_size - 48 - 3) * 2
        pms_padding_hex = ('abcd' * (pad_len // 2 + 1))[:pad_len]
        return pms_padding_hex
    _FINISHED_RECORD = bytearray.fromhex('005091a3b6aaa2b64d126e5583b04c113259c4efa48e40a19b8e5f2542c3b1d30f8d80b7582b72f08b21dfcbff09d4b281676a0fb40d48c20c4f388617ff5c00808a96fbfe9bb6cc631101a6ba6b6bc696f0')

    @classmethod
    def get_finished_record_bytes(cls, tls_version: tls_parser.tls_version.TlsVersionEnum) -> bytes:
        if False:
            while True:
                i = 10
        'The Finished TLS record corresponding to the hardcoded PMS used in the Client Key Exchange record.'
        return b'\x16' + TlsRecordTlsVersionBytes[tls_version.name].value + cls._FINISHED_RECORD

class RobotServerResponsesAnalyzer:

    def __init__(self, payload_responses: Dict[RobotPmsPaddingPayloadEnum, List[str]], attempts_count: int) -> None:
        if False:
            return 10
        for server_responses in payload_responses.values():
            if len(server_responses) != attempts_count:
                raise ValueError()
        self._payload_responses = payload_responses
        self._attempts_count = attempts_count

    def compute_result_enum(self) -> RobotScanResultEnum:
        if False:
            return 10
        "Look at the server's response to each ROBOT payload and return the conclusion of the analysis."
        for (payload_enum, server_responses) in self._payload_responses.items():
            if len(set(server_responses)) != 1:
                return RobotScanResultEnum.UNKNOWN_INCONSISTENT_RESULTS
        if len(set([server_responses[0] for server_responses in self._payload_responses.values()])) == 1:
            return RobotScanResultEnum.NOT_VULNERABLE_NO_ORACLE
        response_1 = self._payload_responses[RobotPmsPaddingPayloadEnum.WRONG_FIRST_TWO_BYTES][0]
        response_2 = self._payload_responses[RobotPmsPaddingPayloadEnum.WRONG_POSITION_00][0]
        response_3 = self._payload_responses[RobotPmsPaddingPayloadEnum.NO_00_IN_THE_MIDDLE][0]
        if response_1 == response_2 == response_3:
            return RobotScanResultEnum.VULNERABLE_WEAK_ORACLE
        else:
            return RobotScanResultEnum.VULNERABLE_STRONG_ORACLE

class ServerDoesNotSupportRsa(Exception):
    pass

def test_robot(server_info: ServerConnectivityInfo) -> Dict[RobotPmsPaddingPayloadEnum, str]:
    if False:
        while True:
            i = 10
    if server_info.tls_probing_result.highest_tls_version_supported.value >= TlsVersionEnum.TLS_1_3.value:
        tls_version_to_use = TlsVersionEnum.TLS_1_2
    else:
        tls_version_to_use = server_info.tls_probing_result.highest_tls_version_supported
    rsa_params = None
    if tls_version_to_use == TlsVersionEnum.TLS_1_2:
        cipher_string = 'AES128-GCM-SHA256:AES256-GCM-SHA384'
        rsa_params = _get_rsa_parameters(server_info, tls_version_to_use, cipher_string)
    if rsa_params is None:
        cipher_string = 'RSA'
        rsa_params = _get_rsa_parameters(server_info, tls_version_to_use, cipher_string)
    if rsa_params is None:
        raise ServerDoesNotSupportRsa()
    rsa_modulus = rsa_params.n
    rsa_exponent = rsa_params.e
    robot_should_complete_handshake = True
    server_responses_per_robot_payloads = _run_oracle_detection(server_info, tls_version_to_use, cipher_string, rsa_modulus, rsa_exponent, robot_should_complete_handshake)
    return server_responses_per_robot_payloads
    robot_result_enum = RobotServerResponsesAnalyzer({payload_enum: [response] for (payload_enum, response) in server_responses_per_robot_payloads.items()}, 1).compute_result_enum()
    if robot_result_enum == RobotScanResultEnum.NOT_VULNERABLE_NO_ORACLE:
        robot_should_complete_handshake = False
        server_responses_per_robot_payloads = _run_oracle_detection(server_info, tls_version_to_use, cipher_string, rsa_modulus, rsa_exponent, robot_should_complete_handshake)
    return server_responses_per_robot_payloads

def _run_oracle_detection(server_info: ServerConnectivityInfo, tls_version_to_use: TlsVersionEnum, cipher_string: str, rsa_modulus: int, rsa_exponent: int, should_complete_handshake: bool) -> Dict[RobotPmsPaddingPayloadEnum, str]:
    if False:
        print('Hello World!')
    server_responses_per_robot_payloads: Dict[RobotPmsPaddingPayloadEnum, str] = {}
    for payload_enum in RobotPmsPaddingPayloadEnum:
        server_response = _send_robot_payload(server_info, tls_version_to_use, cipher_string, payload_enum, should_complete_handshake, rsa_modulus, rsa_exponent)
        server_responses_per_robot_payloads[payload_enum] = server_response
    return server_responses_per_robot_payloads

def _get_rsa_parameters(server_info: ServerConnectivityInfo, tls_version: TlsVersionEnum, openssl_cipher_string: str) -> Optional[RSAPublicNumbers]:
    if False:
        print('Hello World!')
    ssl_connection = server_info.get_preconfigured_tls_connection(override_tls_version=tls_version, should_use_legacy_openssl=True)
    ssl_connection.ssl_client.set_cipher_list(openssl_cipher_string)
    parsed_cert = None
    try:
        ssl_connection.connect()
        cert_as_pem = ssl_connection.ssl_client.get_received_chain()[0]
        parsed_cert = load_pem_x509_certificate(cert_as_pem.encode('ascii'), backend=default_backend())
    except ServerRejectedTlsHandshake:
        pass
    except ClientCertificateRequested:
        raise
    finally:
        ssl_connection.close()
    if parsed_cert:
        public_key = parsed_cert.public_key()
        if isinstance(public_key, RSAPublicKey):
            return public_key.public_numbers()
        else:
            return None
    else:
        return None

def _send_robot_payload(server_info: ServerConnectivityInfo, tls_version_to_use: TlsVersionEnum, rsa_cipher_string: str, robot_payload_enum: RobotPmsPaddingPayloadEnum, robot_should_finish_handshake: bool, rsa_modulus: int, rsa_exponent: int) -> str:
    if False:
        return 10
    ssl_connection = server_info.get_preconfigured_tls_connection(override_tls_version=tls_version_to_use)
    ssl_connection.ssl_client.do_handshake = types.MethodType(do_handshake_with_robot, ssl_connection.ssl_client)
    ssl_connection.ssl_client.set_cipher_list(rsa_cipher_string)
    tls_parser_tls_version: tls_parser.tls_version.TlsVersionEnum
    if tls_version_to_use == TlsVersionEnum.SSL_3_0:
        tls_parser_tls_version = tls_parser.tls_version.TlsVersionEnum.SSLV3
    elif tls_version_to_use == TlsVersionEnum.TLS_1_0:
        tls_parser_tls_version = tls_parser.tls_version.TlsVersionEnum.TLSV1
    elif tls_version_to_use == TlsVersionEnum.TLS_1_1:
        tls_parser_tls_version = tls_parser.tls_version.TlsVersionEnum.TLSV1_1
    elif tls_version_to_use == TlsVersionEnum.TLS_1_2:
        tls_parser_tls_version = tls_parser.tls_version.TlsVersionEnum.TLSV1_2
    else:
        raise ValueError('Should never happen')
    cke_payload = _RobotTlsRecordPayloads.get_client_key_exchange_record(robot_payload_enum, tls_parser_tls_version, rsa_modulus, rsa_exponent)
    ssl_connection.ssl_client._robot_cke_record = cke_payload
    ssl_connection.ssl_client._robot_should_finish_handshake = robot_should_finish_handshake
    server_response = ''
    try:
        ssl_connection.connect()
    except ServerResponseToRobot as e:
        server_response = e.server_response
    except socket.timeout:
        server_response = 'Connection timed out'
    except ServerRejectedTlsHandshake:
        if server_info.tls_probing_result.client_auth_requirement != ClientAuthRequirementEnum.DISABLED:
            raise ClientCertificateRequested(ca_list=[])
        else:
            raise
    finally:
        ssl_connection.close()
    return server_response

class ServerResponseToRobot(Exception):

    def __init__(self, server_response: str) -> None:
        if False:
            return 10
        self.server_response = server_response

def do_handshake_with_robot(self):
    if False:
        return 10
    'Modified do_handshake() to send a ROBOT payload and return the result.'
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
        except NotEnoughData:
            raw_ssl_bytes = self._sock.recv(16381)
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
        self._sock.send(self._robot_cke_record.to_bytes())
        if self._robot_should_finish_handshake:
            ccs_record = TlsChangeCipherSpecRecord.from_parameters(tls_version=tls_parser.tls_version.TlsVersionEnum[self._ssl_version.name])
            self._sock.send(ccs_record.to_bytes())
            finished_record_bytes = _RobotTlsRecordPayloads.get_finished_record_bytes(self._ssl_version)
            self._sock.send(finished_record_bytes)
        while True:
            try:
                (tls_record, len_consumed) = TlsRecordParser.parse_bytes(remaining_bytes)
                remaining_bytes = remaining_bytes[len_consumed:]
            except NotEnoughData:
                try:
                    raw_ssl_bytes = self._sock.recv(16381)
                    if not raw_ssl_bytes:
                        raise ServerResponseToRobot('No data')
                except socket.error as e:
                    raise ServerResponseToRobot('socket.error {}'.format(str(e)))
                remaining_bytes = remaining_bytes + raw_ssl_bytes
                continue
            if isinstance(tls_record, TlsAlertRecord):
                raise ServerResponseToRobot('TLS Alert {} {}'.format(tls_record.alert_description, tls_record.alert_severity))
            else:
                break
        raise ServerResponseToRobot('Ok')