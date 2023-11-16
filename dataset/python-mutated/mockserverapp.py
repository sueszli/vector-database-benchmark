"""Example JupyterServer app subclass"""
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.serverapp import ServerApp
from tornado import web

class TreeHandler(JupyterHandler):

    @web.authenticated
    def get(self):
        if False:
            return 10
        self.write('OK!')

class MockServerApp(ServerApp):

    def initialize(self, argv=None):
        if False:
            print('Hello World!')
        self.default_url = '/tree'
        super().initialize(argv)
        self.web_app.add_handlers('.*$', [(self.base_url + 'tree/?', TreeHandler)])