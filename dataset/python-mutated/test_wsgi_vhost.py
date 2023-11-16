import cherrypy
from cherrypy.test import helper

class WSGI_VirtualHost_Test(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            return 10

        class ClassOfRoot(object):

            def __init__(self, name):
                if False:
                    i = 10
                    return i + 15
                self.name = name

            @cherrypy.expose
            def index(self):
                if False:
                    i = 10
                    return i + 15
                return 'Welcome to the %s website!' % self.name
        default = cherrypy.Application(None)
        domains = {}
        for year in range(1997, 2008):
            app = cherrypy.Application(ClassOfRoot('Class of %s' % year))
            domains['www.classof%s.example' % year] = app
        cherrypy.tree.graft(cherrypy._cpwsgi.VirtualHost(default, domains))

    def test_welcome(self):
        if False:
            print('Hello World!')
        if not cherrypy.server.using_wsgi:
            return self.skip('skipped (not using WSGI)... ')
        for year in range(1997, 2008):
            self.getPage('/', headers=[('Host', 'www.classof%s.example' % year)])
            self.assertBody('Welcome to the Class of %s website!' % year)