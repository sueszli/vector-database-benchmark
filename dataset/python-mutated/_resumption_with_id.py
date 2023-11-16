from enum import Enum
from typing import Optional, Tuple
import nassl
from sslyze.errors import ServerRejectedTlsHandshake
from sslyze.server_connectivity import ServerConnectivityInfo, TlsVersionEnum

class TlsResumptionSupportEnum(str, Enum):
    """The result of attempting to resume TLS sessions with the server.

    Attributes:
        FULLY_SUPPORTED: All the session resumption attempts were successful.
        PARTIALLY_SUPPORTED: Only some of the session resumption attempts were successful.
        NOT_SUPPORTED: None of the session resumption attempts were successful.
        SERVER_IS_TLS_1_3_ONLY: The server only supports TLS 1.3, which does not support Session ID nor TLS Tickets
            resumption.
    """
    FULLY_SUPPORTED = 'FULLY_SUPPORTED'
    PARTIALLY_SUPPORTED = 'PARTIALLY_SUPPORTED'
    NOT_SUPPORTED = 'NOT_SUPPORTED'
    SERVER_IS_TLS_1_3_ONLY = 'SERVER_IS_TLS_1_3_ONLY'

class _ScanJobResultEnum(Enum):
    TLS_TICKET_RESUMPTION = 1
    SESSION_ID_RESUMPTION = 2

class ServerOnlySupportsTls13(Exception):
    """If the server only supports TLS 1.3 or higher, it does not support session resumption with IDs or tickets."""
    pass

def retrieve_tls_session(server_info: ServerConnectivityInfo, session_to_resume: Optional[nassl._nassl.SSL_SESSION]=None, should_enable_tls_ticket: bool=False) -> nassl._nassl.SSL_SESSION:
    if False:
        for i in range(10):
            print('nop')
    'Connect to the server and returns the session object that was assigned for that connection.\n\n    If ssl_session is given, tries to resume that session.\n    '
    if server_info.tls_probing_result.highest_tls_version_supported.value >= TlsVersionEnum.TLS_1_3.value:
        tls_version_to_use = TlsVersionEnum.TLS_1_2
        downgraded_from_tls_1_3 = True
    else:
        tls_version_to_use = server_info.tls_probing_result.highest_tls_version_supported
        downgraded_from_tls_1_3 = False
    ssl_connection = server_info.get_preconfigured_tls_connection(override_tls_version=tls_version_to_use)
    if not should_enable_tls_ticket:
        ssl_connection.ssl_client.disable_stateless_session_resumption()
    if session_to_resume:
        ssl_connection.ssl_client.set_session(session_to_resume)
    try:
        ssl_connection.connect()
        new_session = ssl_connection.ssl_client.get_session()
    except ServerRejectedTlsHandshake:
        if downgraded_from_tls_1_3:
            raise ServerOnlySupportsTls13()
        else:
            raise
    finally:
        ssl_connection.close()
    return new_session

def _extract_session_id(ssl_session: nassl._nassl.SSL_SESSION) -> str:
    if False:
        while True:
            i = 10
    'Extract the SSL session ID from a SSL session object or raises IndexError if the session ID was not set.'
    session_string = ssl_session.as_text().split('Session-ID:')[1]
    session_id = session_string.split('Session-ID-ctx:')[0].strip()
    return session_id

def resume_with_session_id(server_info: ServerConnectivityInfo) -> Tuple[_ScanJobResultEnum, bool]:
    if False:
        while True:
            i = 10
    'Perform one session resumption using Session IDs.'
    session1 = retrieve_tls_session(server_info)
    try:
        session1_id = _extract_session_id(session1)
    except IndexError:
        return (_ScanJobResultEnum.SESSION_ID_RESUMPTION, False)
    if session1_id == '':
        return (_ScanJobResultEnum.SESSION_ID_RESUMPTION, False)
    session2 = retrieve_tls_session(server_info, session_to_resume=session1)
    try:
        session2_id = _extract_session_id(session2)
    except IndexError:
        return (_ScanJobResultEnum.SESSION_ID_RESUMPTION, False)
    if session1_id != session2_id:
        return (_ScanJobResultEnum.SESSION_ID_RESUMPTION, False)
    return (_ScanJobResultEnum.SESSION_ID_RESUMPTION, True)