import http.server as SimpleHTTPServer
import json
import threading
from http.server import HTTPServer
from multiprocessing import Process

class KVHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

    def do_GET(self):
        if False:
            return 10
        with self.server.kv_lock:
            ret = {}
            for (k, v) in self.server.kv.items():
                if k.startswith(self.path):
                    ret[k] = v.decode(encoding='utf-8')
            if ret:
                self.output(200, json.dumps(ret).encode('utf-8'))
            else:
                self.output(404)

    def do_PUT(self):
        if False:
            return 10
        self.do_POST()

    def do_POST(self):
        if False:
            return 10
        content_length = int(self.headers['Content-Length'] or 0)
        try:
            value = self.rfile.read(content_length)
            with self.server.kv_lock:
                self.server.kv[self.path] = value
                self.output(200)
                return
        except:
            self.output(500)

    def do_DELETE(self):
        if False:
            i = 10
            return i + 15
        with self.server.kv_lock:
            if self.path in self.server.kv:
                del self.server.kv[self.path]
                self.output(200)
            else:
                self.output(404)

    def output(self, code, value=''):
        if False:
            for i in range(10):
                print('nop')
        self.send_response(code)
        self.send_header('Content-Length', len(value))
        self.send_header('Content-Type', 'application/json; charset=utf8')
        self.end_headers()
        if value:
            self.wfile.write(value)

    def log_message(self, format, *args):
        if False:
            i = 10
            return i + 15
        return

class KVServer(HTTPServer):

    def __init__(self, port):
        if False:
            while True:
                i = 10
        super().__init__(('', port), KVHandler)
        self.kv_lock = threading.Lock()
        self.kv = {'/healthy': b'ok'}
        self.port = port
        self.stopped = False
        self.started = False

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        self.listen_thread = threading.Thread(target=self.serve_forever)
        self.listen_thread.start()
        self.started = True

    def stop(self):
        if False:
            return 10
        self.shutdown()
        self.listen_thread.join()
        self.server_close()
        self.stopped = True

class PKVServer:

    def __init__(self, port):
        if False:
            return 10
        self._server = KVServer(port)

    def start(self):
        if False:
            while True:
                i = 10
        self.proc = Process(target=self._server.start)
        self.proc.daemon = True
        self.proc.start()

    def stop(self):
        if False:
            print('Hello World!')
        self._server.stop()
        self.proc.join()

    @property
    def started(self):
        if False:
            return 10
        return self._server.started

    @property
    def stopped(self):
        if False:
            while True:
                i = 10
        return self._server.stopped
if __name__ == '__main__':
    kv = KVServer(8090)
    kv.start()
    import time
    time.sleep(600)