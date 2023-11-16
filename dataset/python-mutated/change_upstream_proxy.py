from mitmproxy import http
from mitmproxy.connection import Server
from mitmproxy.net.server_spec import ServerSpec

def proxy_address(flow: http.HTTPFlow) -> tuple[str, int]:
    if False:
        i = 10
        return i + 15
    if hash(flow.request.host) % 2 == 1:
        return ('localhost', 8082)
    else:
        return ('localhost', 8081)

def request(flow: http.HTTPFlow) -> None:
    if False:
        return 10
    address = proxy_address(flow)
    is_proxy_change = address != flow.server_conn.via.address
    server_connection_already_open = flow.server_conn.timestamp_start is not None
    if is_proxy_change and server_connection_already_open:
        flow.server_conn = Server(address=flow.server_conn.address)
    flow.server_conn.via = ServerSpec('http', address)