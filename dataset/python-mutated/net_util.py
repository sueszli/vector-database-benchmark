import socket
from typing import Optional
from typing_extensions import Final
from streamlit import util
from streamlit.logger import get_logger
LOGGER = get_logger(__name__)
_AWS_CHECK_IP: Final = 'http://checkip.amazonaws.com'
_external_ip: Optional[str] = None
_internal_ip: Optional[str] = None

def get_external_ip() -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    'Get the *external* IP address of the current machine.\n\n    Returns\n    -------\n    string\n        The external IPv4 address of the current machine.\n\n    '
    global _external_ip
    if _external_ip is not None:
        return _external_ip
    response = _make_blocking_http_get(_AWS_CHECK_IP, timeout=5)
    if _looks_like_an_ip_adress(response):
        _external_ip = response
    else:
        LOGGER.warning('Did not auto detect external IP.\nPlease go to %s for debugging hints.', util.HELP_DOC)
        _external_ip = None
    return _external_ip

def get_internal_ip() -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    'Get the *local* IP address of the current machine.\n\n    From: https://stackoverflow.com/a/28950776\n\n    Returns\n    -------\n    string\n        The local IPv4 address of the current machine.\n\n    '
    global _internal_ip
    if _internal_ip is not None:
        return _internal_ip
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(('8.8.8.8', 1))
            _internal_ip = s.getsockname()[0]
        except Exception:
            _internal_ip = '127.0.0.1'
    return _internal_ip

def _make_blocking_http_get(url: str, timeout: float=5) -> Optional[str]:
    if False:
        while True:
            i = 10
    import requests
    try:
        text = requests.get(url, timeout=timeout).text
        if isinstance(text, str):
            text = text.strip()
        return text
    except Exception:
        return None

def _looks_like_an_ip_adress(address: Optional[str]) -> bool:
    if False:
        i = 10
        return i + 15
    if address is None:
        return False
    try:
        socket.inet_pton(socket.AF_INET, address)
        return True
    except (AttributeError, OSError):
        pass
    try:
        socket.inet_pton(socket.AF_INET6, address)
        return True
    except (AttributeError, OSError):
        pass
    return False