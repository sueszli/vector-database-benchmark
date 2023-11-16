from reactpy import run
from reactpy.backend import tornado as tornado_server
tornado_server.configure = lambda _, cmpt: run(cmpt)
import tornado.ioloop
import tornado.web
from reactpy import component, html
from reactpy.backend.tornado import configure

@component
def HelloWorld():
    if False:
        while True:
            i = 10
    return html.h1('Hello, world!')

def make_app():
    if False:
        for i in range(10):
            print('nop')
    app = tornado.web.Application()
    configure(app, HelloWorld)
    return app
if __name__ == '__main__':
    app = make_app()
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()