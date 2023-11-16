"""
Tutorial - Multiple methods

This tutorial shows you how to link to other methods of your request
handler.
"""
import os.path
import cherrypy

class HelloWorld:

    @cherrypy.expose
    def index(self):
        if False:
            for i in range(10):
                print('nop')
        return 'We have an <a href="show_msg">important message</a> for you!'

    @cherrypy.expose
    def show_msg(self):
        if False:
            i = 10
            return i + 15
        return 'Hello world!'
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld(), config=tutconf)