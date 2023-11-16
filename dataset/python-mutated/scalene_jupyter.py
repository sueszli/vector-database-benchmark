import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Any, Optional

class ScaleneJupyter:

    @staticmethod
    def find_available_port(start_port: int, end_port: int) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds an available port within a given range.\n\n        Parameters:\n        - start_port (int): the starting port number to search from\n        - end_port (int): the ending port number to search up to (inclusive)\n\n        Returns:\n        - int: the first available port number found in the given range, or None if no ports are available\n        '
        for port in range(start_port, end_port + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None

    @staticmethod
    def display_profile(port: int, profile_fname: str) -> None:
        if False:
            i = 10
            return i + 15
        from IPython.core.display import display
        from IPython.display import IFrame

        class RequestHandler(BaseHTTPRequestHandler):

            def _send_response(self, content: str) -> None:
                if False:
                    print('Hello World!')
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(bytes(content, 'utf8'))

            def log_message(self, format: str, *args: Any) -> None:
                if False:
                    return 10
                'overriding log_message to disable all messages from webserver'
                pass

            def do_GET(self) -> None:
                if False:
                    while True:
                        i = 10
                if self.path == '/':
                    try:
                        with open(profile_fname) as f:
                            content = f.read()
                        self._send_response(content)
                    except FileNotFoundError:
                        print('Scalene error: profile file not found.')
                elif self.path == '/shutdown':
                    self.server.should_shutdown = True
                    self.send_response(204)
                else:
                    self.send_response(404)

        class MyHTTPServer(HTTPServer):
            """Redefine to check `should_shutdown` flag."""

            def serve_forever(self, poll_interval: float=0.5) -> None:
                if False:
                    return 10
                self.should_shutdown = False
                while not self.should_shutdown:
                    self.handle_request()

        class local_server:

            def run_server(self) -> None:
                if False:
                    i = 10
                    return i + 15
                try:
                    server_address = ('', port)
                    self.httpd = MyHTTPServer(server_address, RequestHandler)
                    self.httpd.serve_forever()
                except BaseException as be:
                    print('server failure', be)
                    pass
        the_server = local_server()
        server_thread = Thread(target=the_server.run_server)
        server_thread.start()
        display(IFrame(src=f'http://localhost:{port}', width='100%', height='400'))
        Thread(target=lambda : server_thread.join()).start()
        import time
        time.sleep(2)
        import sys
        sys.exit()