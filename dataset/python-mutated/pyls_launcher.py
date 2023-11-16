import logging
import subprocess
import threading
import os
from tornado import ioloop, process, web, websocket
from pylsp_jsonrpc import streams
try:
    import ujson as json
except Exception:
    import json
log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))

class LanguageServerWebSocketHandler(websocket.WebSocketHandler):
    """Setup tornado websocket handler to host an external language server."""
    writer = None
    id = None
    proc = None
    loop = None

    def open(self):
        if False:
            i = 10
            return i + 15
        self.id = str(self)
        log.info('Spawning pylsp subprocess' + self.id)
        self.proc = process.Subprocess(self.procargs, env=os.environ, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.writer = streams.JsonRpcStreamWriter(self.proc.stdin)

        def consume():
            if False:
                for i in range(10):
                    print('nop')
            self.loop = ioloop.IOLoop()
            reader = streams.JsonRpcStreamReader(self.proc.stdout)

            def on_listen(msg):
                if False:
                    while True:
                        i = 10
                try:
                    self.write_message(json.dumps(msg))
                except Exception as e:
                    log.error('Error writing message', e)
            reader.listen(on_listen)
        self.thread = threading.Thread(target=consume)
        self.thread.daemon = True
        self.thread.start()

    def on_message(self, message):
        if False:
            print('Hello World!')
        'Forward client->server messages to the endpoint.'
        if not 'Unhandled method' in message:
            self.writer.write(json.loads(message))

    def on_close(self) -> None:
        if False:
            while True:
                i = 10
        log.info('CLOSING: ' + str(self.id))
        self.proc.proc.terminate()
        self.writer.close()
        self.loop.stop()

    def check_origin(self, origin):
        if False:
            while True:
                i = 10
        return True

class PyrightLS(LanguageServerWebSocketHandler):
    procargs = ['pipenv', 'run', 'pyright-langserver', '--stdio']

class DiagnosticLS(LanguageServerWebSocketHandler):
    procargs = ['diagnostic-languageserver', '--stdio', '--log-level', '4']

class RuffLS(LanguageServerWebSocketHandler):
    procargs = ['ruff-lsp']

class DenoLS(LanguageServerWebSocketHandler):
    procargs = ['deno', 'lsp']

class BunLS(LanguageServerWebSocketHandler):
    procargs = ['typescript-language-server', '--stdio']

class GoLS(LanguageServerWebSocketHandler):
    procargs = ['gopls', 'serve']

class MainHandler(web.RequestHandler):

    def get(self):
        if False:
            return 10
        self.write('ok')
if __name__ == '__main__':
    monaco_path = '/tmp/monaco'
    os.makedirs(monaco_path, exist_ok=True)
    print('The monaco directory is created!')
    go_mod_path = os.path.join(monaco_path, 'go.mod')
    if not os.path.exists(go_mod_path):
        f = open(go_mod_path, 'w')
        f.write('module mymod\ngo 1.19')
        f.close()
    port = int(os.environ.get('PORT', '3001'))
    app = web.Application([('/ws/pyright', PyrightLS), ('/ws/diagnostic', DiagnosticLS), ('/ws/ruff', RuffLS), ('/ws/deno', DenoLS), ('/ws/bun', BunLS), ('/ws/go', GoLS), ('/', MainHandler), ('/health', MainHandler)])
    app.listen(port, address='0.0.0.0')
    ioloop.IOLoop.current().start()