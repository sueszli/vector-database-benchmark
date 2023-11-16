"""
Tutorial - Hello World

The most basic (working) CherryPy application possible.
"""
import os.path
import cherrypy

class HelloWorld:
    """ Sample request handler class. """

    @cherrypy.expose
    def index(self):
        if False:
            while True:
                i = 10
        return 'Hello world!'
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld(), config=tutconf)