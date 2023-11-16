import os
from typing import Dict, Union
from tornado import ioloop
from tornado.escape import json_decode
from tornado.web import Application, RequestHandler, StaticFileHandler, url
from tornado.websocket import WebSocketHandler
from reactivex import operators as ops
from reactivex.subject import Subject
(UP, DOWN, LEFT, RIGHT, B, A) = (38, 40, 37, 39, 66, 65)
codes = [UP, UP, DOWN, DOWN, LEFT, RIGHT, LEFT, RIGHT, B, A]

class WSHandler(WebSocketHandler):

    def open(self):
        if False:
            while True:
                i = 10
        print('WebSocket opened')
        self.subject: Subject[Dict[str, int]] = Subject()
        query = self.subject.pipe(ops.map(lambda obj: obj['keycode']), ops.window_with_count(10, 1), ops.flat_map(lambda win: win.pipe(ops.sequence_equal(codes))), ops.filter(lambda equal: equal))
        query.subscribe(on_next=lambda x: self.write_message('Konami!'))

    def on_message(self, message: Union[str, bytes]):
        if False:
            print('Hello World!')
        obj = json_decode(message)
        self.subject.on_next(obj)

    def on_close(self):
        if False:
            return 10
        print('WebSocket closed')

class MainHandler(RequestHandler):

    def get(self):
        if False:
            print('Hello World!')
        self.render('index.html')

def main():
    if False:
        while True:
            i = 10
    port = os.environ.get('PORT', 8080)
    app = Application([url('/', MainHandler), ('/ws', WSHandler), ('/static/(.*)', StaticFileHandler, {'path': '.'})])
    print('Starting server at port: %s' % port)
    app.listen(int(port))
    ioloop.IOLoop.current().start()
if __name__ == '__main__':
    main()