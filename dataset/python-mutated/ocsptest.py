from __future__ import annotations
import argparse
import logging
import socket
from pymongo.pyopenssl_context import SSLContext
from pymongo.ssl_support import get_ssl_context
FORMAT = '%(asctime)s %(levelname)s %(module)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

def check_ocsp(host: str, port: int, capath: str) -> None:
    if False:
        return 10
    ctx = get_ssl_context(None, None, capath, None, False, False, False)
    assert isinstance(ctx, SSLContext)
    s = socket.socket()
    s.connect((host, port))
    try:
        s = ctx.wrap_socket(s, server_hostname=host)
    finally:
        s.close()

def main() -> None:
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Debug OCSP')
    parser.add_argument('--host', type=str, required=True, help='Host to connect to')
    parser.add_argument('-p', '--port', type=int, default=443, help='Port to connect to')
    parser.add_argument('--ca_file', type=str, default=None, help='CA file for host')
    args = parser.parse_args()
    check_ocsp(args.host, args.port, args.ca_file)
if __name__ == '__main__':
    main()