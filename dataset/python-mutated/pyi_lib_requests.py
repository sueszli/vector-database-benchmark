import socket
try:
    import BaseHTTPServer
    import SimpleHTTPServer
except ImportError:
    import http.server as BaseHTTPServer
    import http.server as SimpleHTTPServer
import os
import ssl
import sys
import threading
import time
import requests
'\nNote: to re-create the server.pem file use the following commands:\n\ncd /path/to/pyinstaller.git/tests/functional\nopenssl req -new -x509 -keyout data/requests/server.pem     -text -out data/requests/server.pem -days 36500     -nodes -config data/requests/openssl.conf\n'
if getattr(sys, 'frozen', False):
    basedir = sys._MEIPASS
else:
    basedir = os.path.dirname(__file__)
SERVER_CERT = os.path.join(basedir, 'server.pem')
if not os.path.exists(SERVER_CERT):
    raise SystemExit('Certificate-File %s is missing' % SERVER_CERT)

def main():
    if False:
        i = 10
        return i + 15
    SERVER_PORT = 8443
    httpd = None
    while SERVER_PORT < 8493:
        try:
            httpd = BaseHTTPServer.HTTPServer(('localhost', SERVER_PORT), SimpleHTTPServer.SimpleHTTPRequestHandler)
        except socket.error as e:
            if e.errno == 98:
                SERVER_PORT += 1
                continue
            else:
                raise
        else:
            break
    else:
        assert False, 'Could not bind server port: all ports in use.'
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile=SERVER_CERT, keyfile=None)
    httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)

    def ssl_server():
        if False:
            for i in range(10):
                print('nop')
        httpd.serve_forever()
    thread = threading.Thread(target=ssl_server)
    thread.daemon = True
    thread.start()
    time.sleep(1)
    requests.get('https://localhost:{}'.format(SERVER_PORT), verify=SERVER_CERT)
if __name__ == '__main__':
    main()