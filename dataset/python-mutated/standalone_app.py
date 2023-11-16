import multiprocessing
import gunicorn.app.base

def number_of_workers():
    if False:
        return 10
    return multiprocessing.cpu_count() * 2 + 1

def handler_app(environ, start_response):
    if False:
        while True:
            i = 10
    response_body = b'Works fine'
    status = '200 OK'
    response_headers = [('Content-Type', 'text/plain')]
    start_response(status, response_headers)
    return [response_body]

class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        if False:
            for i in range(10):
                print('nop')
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = {key: value for (key, value) in self.options.items() if key in self.cfg.settings and value is not None}
        for (key, value) in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        if False:
            return 10
        return self.application
if __name__ == '__main__':
    options = {'bind': '%s:%s' % ('127.0.0.1', '8080'), 'workers': number_of_workers()}
    StandaloneApplication(handler_app, options).run()