from __future__ import annotations
import sys
import typing
import tornado.httpserver
import tornado.ioloop
import tornado.web
from dummyserver.proxy import ProxyHandler
from dummyserver.tornadoserver import DEFAULT_CERTS, ssl_options_to_context

def run_proxy(port: int, certs: dict[str, typing.Any]=DEFAULT_CERTS) -> None:
    if False:
        while True:
            i = 10
    "\n    Run proxy on the specified port using the provided certs.\n\n    Example usage:\n\n    python -m dummyserver.https_proxy\n\n    You'll need to ensure you have access to certain packages such as trustme,\n    tornado, urllib3.\n    "
    upstream_ca_certs = certs.get('ca_certs')
    app = tornado.web.Application([('.*', ProxyHandler)], upstream_ca_certs=upstream_ca_certs)
    ssl_opts = ssl_options_to_context(**certs)
    http_server = tornado.httpserver.HTTPServer(app, ssl_options=ssl_opts)
    http_server.listen(port)
    ioloop = tornado.ioloop.IOLoop.instance()
    try:
        ioloop.start()
    except KeyboardInterrupt:
        ioloop.stop()
if __name__ == '__main__':
    port = 8443
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    print(f'Starting HTTPS proxy on port {port}')
    run_proxy(port)