"""
Tutorial - Passing variables

This tutorial shows you how to pass GET/POST variables to methods.
"""
import os.path
import cherrypy

class WelcomePage:

    @cherrypy.expose
    def index(self):
        if False:
            print('Hello World!')
        return '\n            <form action="greetUser" method="GET">\n            What is your name?\n            <input type="text" name="name" />\n            <input type="submit" />\n            </form>'

    @cherrypy.expose
    def greetUser(self, name=None):
        if False:
            print('Hello World!')
        if name:
            return "Hey %s, what's up?" % name
        elif name is None:
            return 'Please enter your name <a href="./">here</a>.'
        else:
            return 'No, really, enter your name <a href="./">here</a>.'
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(WelcomePage(), config=tutconf)