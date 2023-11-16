"""
Local test server based on http.server
"""
import contextlib
import http.server
import queue
import socket
import threading

def run_test_server(directory: str) -> http.server.ThreadingHTTPServer:
    if False:
        while True:
            i = 10
    '\n    Run a test server on a random port. Inspect returned server to get port,\n    shutdown etc.\n    '

    class DualStackServer(http.server.ThreadingHTTPServer):
        daemon_threads = False
        allow_reuse_address = True
        request_queue_size = 64

        def server_bind(self):
            if False:
                print('Hello World!')
            with contextlib.suppress(Exception):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            return super().server_bind()

        def finish_request(self, request, client_address):
            if False:
                return 10
            self.RequestHandlerClass(request, client_address, self, directory=directory)

    def start_server(queue):
        if False:
            print('Hello World!')
        with DualStackServer(('127.0.0.1', 0), http.server.SimpleHTTPRequestHandler) as httpd:
            (host, port) = httpd.socket.getsockname()[:2]
            queue.put(httpd)
            url_host = f'[{host}]' if ':' in host else host
            print(f'Serving HTTP on {host} port {port} (http://{url_host}:{port}/) ...')
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print('\nKeyboard interrupt received, exiting.')
    started = queue.Queue()
    threading.Thread(target=start_server, args=(started,), daemon=True).start()
    return started.get(timeout=1)
if __name__ == '__main__':
    server = run_test_server(directory='.')
    print(server)