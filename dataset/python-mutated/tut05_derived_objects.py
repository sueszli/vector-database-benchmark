"""
Tutorial - Object inheritance

You are free to derive your request handler classes from any base
class you wish. In most real-world applications, you will probably
want to create a central base class used for all your pages, which takes
care of things like printing a common page header and footer.
"""
import os.path
import cherrypy

class Page:
    title = 'Untitled Page'

    def header(self):
        if False:
            i = 10
            return i + 15
        return '\n            <html>\n            <head>\n                <title>%s</title>\n            <head>\n            <body>\n            <h2>%s</h2>\n        ' % (self.title, self.title)

    def footer(self):
        if False:
            for i in range(10):
                print('nop')
        return '\n            </body>\n            </html>\n        '

class HomePage(Page):
    title = 'Tutorial 5'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.another = AnotherPage()

    @cherrypy.expose
    def index(self):
        if False:
            print('Hello World!')
        return self.header() + '\n            <p>\n            Isn\'t this exciting? There\'s\n            <a href="./another/">another page</a>, too!\n            </p>\n        ' + self.footer()

class AnotherPage(Page):
    title = 'Another Page'

    @cherrypy.expose
    def index(self):
        if False:
            for i in range(10):
                print('nop')
        return self.header() + '\n            <p>\n            And this is the amazing second page!\n            </p>\n        ' + self.footer()
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(HomePage(), config=tutconf)