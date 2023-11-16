import socket
import sys
import time
from contextlib import closing
import paddle
__all__ = []

def wait_server_ready(endpoints):
    if False:
        i = 10
        return i + 15
    '\n    Wait until parameter servers are ready, use connext_ex to detect\n    port readiness.\n\n    Args:\n    endpoints (list|tuple): endpoints string list, like:\n    ["127.0.0.1:8080", "127.0.0.1:8081"]\n\n    Examples:\n        .. code-block:: python\n\n             >>> wait_server_ready(["127.0.0.1:8080", "127.0.0.1:8081"])\n    '
    try:
        use_new_comm = paddle.get_flags('FLAGS_dynamic_static_unified_comm')['FLAGS_dynamic_static_unified_comm']
    except:
        use_new_comm = False
    if use_new_comm:
        return
    assert not isinstance(endpoints, str)
    while True:
        all_ok = True
        not_ready_endpoints = []
        for ep in endpoints:
            ip_port = ep.split(':')
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.settimeout(2)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if hasattr(socket, 'SO_REUSEPORT'):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                if result != 0:
                    all_ok = False
                    not_ready_endpoints.append(ep)
        if not all_ok:
            sys.stderr.write('server not ready, wait 3 sec to retry...\n')
            sys.stderr.write('not ready endpoints:' + str(not_ready_endpoints) + '\n')
            sys.stderr.flush()
            time.sleep(3)
        else:
            break