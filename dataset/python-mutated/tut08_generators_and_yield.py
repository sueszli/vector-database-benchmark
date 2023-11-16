"""
Bonus Tutorial: Using generators to return result bodies

Instead of returning a complete result string, you can use the yield
statement to return one result part after another. This may be convenient
in situations where using a template package like CherryPy or Cheetah
would be overkill, and messy string concatenation too uncool. ;-)
"""
import os.path
import cherrypy

class GeneratorDemo:

    def header(self):
        if False:
            while True:
                i = 10
        return '<html><body><h2>Generators rule!</h2>'

    def footer(self):
        if False:
            print('Hello World!')
        return '</body></html>'

    @cherrypy.expose
    def index(self):
        if False:
            i = 10
            return i + 15
        users = ['Remi', 'Carlos', 'Hendrik', 'Lorenzo Lamas']
        yield self.header()
        yield '<h3>List of users:</h3>'
        for user in users:
            yield ('%s<br/>' % user)
        yield self.footer()
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(GeneratorDemo(), config=tutconf)