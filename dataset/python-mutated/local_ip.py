from __future__ import annotations
import socket

def get_local_ip(reachable: str='8.8.8.8') -> str | None:
    if False:
        while True:
            i = 10
    "\n    Get the default local outgoing IPv4 address without sending any packets.\n    This will fail if the target address is known to be unreachable.\n    We use Google DNS's IPv4 address as the default.\n    "
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((reachable, 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()

def get_local_ip6(reachable: str='2001:4860:4860::8888') -> str | None:
    if False:
        print('Hello World!')
    "\n    Get the default local outgoing IPv6 address without sending any packets.\n    This will fail if the target address is known to be unreachable.\n    We use Google DNS's IPv6 address as the default.\n    "
    s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    try:
        s.connect((reachable, 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()