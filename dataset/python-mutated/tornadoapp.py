from datetime import timedelta
from tornado.web import Application, RequestHandler, asynchronous
from tornado.ioloop import IOLoop

class MainHandler(RequestHandler):

    def get(self):
        if False:
            i = 10
            return i + 15
        self.write('Hello, world')

class LongPollHandler(RequestHandler):

    @asynchronous
    def get(self):
        if False:
            return 10
        lines = ['line 1\n', 'line 2\n']

        def send():
            if False:
                print('Hello World!')
            try:
                self.write(lines.pop(0))
                self.flush()
            except:
                self.finish()
            else:
                IOLoop.instance().add_timeout(timedelta(0, 20), send)
        send()
app = Application([('/', MainHandler), ('/longpoll', LongPollHandler)])