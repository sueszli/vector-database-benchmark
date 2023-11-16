"""
Tutorial - Multiple objects

This tutorial shows you how to create a site structure through multiple
possibly nested request handler objects.
"""
import os.path
import cherrypy

class HomePage:

    @cherrypy.expose
    def index(self):
        if False:
            i = 10
            return i + 15
        return '\n            <p>Hi, this is the home page! Check out the other\n            fun stuff on this site:</p>\n\n            <ul>\n                <li><a href="/joke/">A silly joke</a></li>\n                <li><a href="/links/">Useful links</a></li>\n            </ul>'

class JokePage:

    @cherrypy.expose
    def index(self):
        if False:
            i = 10
            return i + 15
        return '\n            <p>"In Python, how do you create a string of random\n            characters?" -- "Read a Perl file!"</p>\n            <p>[<a href="../">Return</a>]</p>'

class LinksPage:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.extra = ExtraLinksPage()

    @cherrypy.expose
    def index(self):
        if False:
            i = 10
            return i + 15
        return '\n            <p>Here are some useful links:</p>\n\n            <ul>\n                <li>\n                    <a href="http://www.cherrypy.dev">The CherryPy Homepage</a>\n                </li>\n                <li>\n                    <a href="http://www.python.org">The Python Homepage</a>\n                </li>\n            </ul>\n\n            <p>You can check out some extra useful\n            links <a href="./extra/">here</a>.</p>\n\n            <p>[<a href="../">Return</a>]</p>\n        '

class ExtraLinksPage:

    @cherrypy.expose
    def index(self):
        if False:
            print('Hello World!')
        return '\n            <p>Here are some extra useful links:</p>\n\n            <ul>\n                <li><a href="http://del.icio.us">del.icio.us</a></li>\n                <li><a href="http://www.cherrypy.dev">CherryPy</a></li>\n            </ul>\n\n            <p>[<a href="../">Return to links page</a>]</p>'
root = HomePage()
root.joke = JokePage()
root.links = LinksPage()
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(root, config=tutconf)