import cherrypy

class Root(object):

    @cherrypy.expose
    def index(self):
        if False:
            while True:
                i = 10
        return 'Hello World!'
cherrypy.config.update({'environment': 'embedded'})
app = cherrypy.tree.mount(Root())