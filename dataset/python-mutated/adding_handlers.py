"""
Example that demonstrates how to add cusom Tornado handlers, for
instance to serve a custom HTML page, or a (REST) api from the same
process.

For the sake of the example, we alo serve two example web apps.

Goto:

* http://localhost:port/ to see a list of served web apps
* http://localhost:port/about to see a the custom about page
* http://localhost:port/api/foo/bar to see the echo api in action


"""
from flexx import flx
from flexxamples.demos.drawing import Drawing
from flexxamples.demos.chatroom import ChatRoom
import tornado.web
flx.serve(Drawing)
flx.serve(ChatRoom)

class MyAboutHandler(tornado.web.RequestHandler):

    def get(self):
        if False:
            return 10
        self.write('<html>This is just an <i>example</i>.</html>')

class MyAPIHandler(tornado.web.RequestHandler):

    def get(self, path):
        if False:
            for i in range(10):
                print('nop')
        self.write('echo ' + path)
tornado_app = flx.current_server().app
tornado_app.add_handlers('.*', [('/about', MyAboutHandler), ('/api/(.*)', MyAPIHandler)])
flx.start()