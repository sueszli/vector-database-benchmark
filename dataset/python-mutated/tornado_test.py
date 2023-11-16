import tornado.ioloop
import tornado.web
import json

class TextHandler(tornado.web.RequestHandler):

    def get(self):
        if False:
            return 10
        self.write('Hello, world!')
application = tornado.web.Application([('/text', TextHandler)])
if __name__ == '__main__':
    application.listen(8000)
    tornado.ioloop.IOLoop.current().start()