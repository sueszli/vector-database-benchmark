import cherrypy

class Root(object):

    @cherrypy.expose
    def text(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Hello, world!'
app = cherrypy.tree.mount(Root())
cherrypy.log.screen = False