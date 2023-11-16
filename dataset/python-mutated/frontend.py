"""A simple web server which responds to HTTP GET requests by consuming CPU.
This binary runs in a GCE VM. It serves HTTP requests on port 80. Every request
with path '/service' consumes 1 core-second of CPU time, with the timeout of
5 (walltime) seconds. The purpose of this application is to demonstrate how
Google Compute Engine Autoscaler can scale a web frontend server based on CPU
utilization.
The original version of this file is available here:
https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/compute/
    autoscaler/demo/tests/test_frontend.py
"""
try:
    import BaseHTTPServer
    import SocketServer
except ImportError:
    import http.server as BaseHTTPServer
    import socketserver as SocketServer
from multiprocessing import Process
import os
import sys
import time
REQUEST_CPUTIME_SEC = 1.0
REQUEST_TIMEOUT_SEC = 5.0

class CpuBurner(object):

    def get_walltime(self):
        if False:
            return 10
        return time.time()

    def get_user_cputime(self):
        if False:
            print('Hello World!')
        return os.times()[0]

    def busy_wait(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(100000):
            pass

    def burn_cpu(self):
        if False:
            while True:
                i = 10
        'Consume REQUEST_CPUTIME_SEC core seconds.\n        This method consumes REQUEST_CPUTIME_SEC core seconds. If unable to\n        complete within REQUEST_TIMEOUT_SEC walltime seconds, it times out and\n        terminates the process.\n        '
        start_walltime_sec = self.get_walltime()
        start_cputime_sec = self.get_user_cputime()
        while self.get_user_cputime() < start_cputime_sec + REQUEST_CPUTIME_SEC:
            self.busy_wait()
            if self.get_walltime() > start_walltime_sec + REQUEST_TIMEOUT_SEC:
                sys.exit(1)

    def handle_http_request(self):
        if False:
            while True:
                i = 10
        'Process a request to consume CPU and produce an HTTP response.'
        start_time = self.get_walltime()
        p = Process(target=self.burn_cpu)
        p.start()
        p.join(timeout=REQUEST_TIMEOUT_SEC + 1)
        if p.is_alive():
            p.terminate()
        if p.exitcode != 0:
            return (500, 'Request failed\n')
        else:
            end_time = self.get_walltime()
            response = 'Request took %.2f walltime seconds\n' % (end_time - start_time)
            return (200, response)

class DemoRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    """Request handler for Demo http server."""

    def do_GET(self):
        if False:
            return 10
        'Handle an HTTP GET request.'
        mapping = {'/': lambda : (200, 'OK'), '/service': CpuBurner().handle_http_request}
        if self.path not in mapping:
            self.send_response(404)
            self.end_headers()
            return
        (code, response) = mapping[self.path]()
        self.send_response(code)
        self.end_headers()
        self.wfile.write(response)
        self.wfile.close()

class DemoHttpServer(SocketServer.ThreadingMixIn, BaseHTTPServer.HTTPServer):
    pass
if __name__ == '__main__':
    httpd = DemoHttpServer(('', 80), DemoRequestHandler)
    httpd.serve_forever()