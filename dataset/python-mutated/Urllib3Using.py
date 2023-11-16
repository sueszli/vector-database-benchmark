from __future__ import print_function
import json
import os
import socket
import ssl
from threading import Thread
import urllib3
if str is bytes:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
else:
    from http.server import BaseHTTPRequestHandler, HTTPServer
started = False

def runHTTPServer():
    if False:
        while True:
            i = 10

    class myServer(BaseHTTPRequestHandler):

        def do_GET(self):
            if False:
                return 10
            if self.path == '/':
                self.path = '/index.html'
            try:
                file_to_open = open(self.path[1:], 'rb').read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(file_to_open)
            except IOError:
                self.send_response(404)
                self.end_headers()

        def log_request(self, code):
            if False:
                print('Hello World!')
            pass
    global port
    global server
    for port in range(8020, 9000):
        server_address = ('127.0.0.1', port)
        try:
            server = HTTPServer(server_address, myServer)
        except OSError:
            continue
        else:
            break
    global started
    started = True
    server.serve_forever()
Thread(target=runHTTPServer).start()
while not started:
    pass
print('Server started.')
http = urllib3.PoolManager()
r = http.request('GET', 'http://localhost:%d/' % port)
print(r.status, r.data)
with open('testjson.json', 'w') as f:
    f.write('{"origin": "some, value"}')
r = http.request('GET', 'http://localhost:%d/testjson.json' % port)
data = json.loads(r.data.decode('utf8'))
if 'Date' in data:
    del data['Date']
print('DATA:', data)
os.remove('testjson.json')
server.shutdown()
print('Server shutdown')
if False:
    hostname = 'www.google.com'
    context = ssl.create_default_context()
    with socket.create_connection((hostname, 443)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            print(ssock.version())
print('OK.')