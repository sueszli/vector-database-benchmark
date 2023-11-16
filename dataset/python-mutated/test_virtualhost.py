import os
import cherrypy
from cherrypy.test import helper
curdir = os.path.join(os.getcwd(), os.path.dirname(__file__))

class VirtualHostTest(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            for i in range(10):
                print('nop')

        class Root:

            @cherrypy.expose
            def index(self):
                if False:
                    i = 10
                    return i + 15
                return 'Hello, world'

            @cherrypy.expose
            def dom4(self):
                if False:
                    i = 10
                    return i + 15
                return 'Under construction'

            @cherrypy.expose
            def method(self, value):
                if False:
                    print('Hello World!')
                return 'You sent %s' % value

        class VHost:

            def __init__(self, sitename):
                if False:
                    for i in range(10):
                        print('nop')
                self.sitename = sitename

            @cherrypy.expose
            def index(self):
                if False:
                    i = 10
                    return i + 15
                return 'Welcome to %s' % self.sitename

            @cherrypy.expose
            def vmethod(self, value):
                if False:
                    i = 10
                    return i + 15
                return 'You sent %s' % value

            @cherrypy.expose
            def url(self):
                if False:
                    for i in range(10):
                        print('nop')
                return cherrypy.url('nextpage')
            static = cherrypy.tools.staticdir.handler(section='/static', dir=curdir)
        root = Root()
        root.mydom2 = VHost('Domain 2')
        root.mydom3 = VHost('Domain 3')
        hostmap = {'www.mydom2.com': '/mydom2', 'www.mydom3.com': '/mydom3', 'www.mydom4.com': '/dom4'}
        cherrypy.tree.mount(root, config={'/': {'request.dispatch': cherrypy.dispatch.VirtualHost(**hostmap)}, '/mydom2/static2': {'tools.staticdir.on': True, 'tools.staticdir.root': curdir, 'tools.staticdir.dir': 'static', 'tools.staticdir.index': 'index.html'}})

    def testVirtualHost(self):
        if False:
            return 10
        self.getPage('/', [('Host', 'www.mydom1.com')])
        self.assertBody('Hello, world')
        self.getPage('/mydom2/', [('Host', 'www.mydom1.com')])
        self.assertBody('Welcome to Domain 2')
        self.getPage('/', [('Host', 'www.mydom2.com')])
        self.assertBody('Welcome to Domain 2')
        self.getPage('/', [('Host', 'www.mydom3.com')])
        self.assertBody('Welcome to Domain 3')
        self.getPage('/', [('Host', 'www.mydom4.com')])
        self.assertBody('Under construction')
        self.getPage('/method?value=root')
        self.assertBody('You sent root')
        self.getPage('/vmethod?value=dom2+GET', [('Host', 'www.mydom2.com')])
        self.assertBody('You sent dom2 GET')
        self.getPage('/vmethod', [('Host', 'www.mydom3.com')], method='POST', body='value=dom3+POST')
        self.assertBody('You sent dom3 POST')
        self.getPage('/vmethod/pos', [('Host', 'www.mydom3.com')])
        self.assertBody('You sent pos')
        self.getPage('/url', [('Host', 'www.mydom2.com')])
        self.assertBody('%s://www.mydom2.com/nextpage' % self.scheme)

    def test_VHost_plus_Static(self):
        if False:
            print('Hello World!')
        self.getPage('/static/style.css', [('Host', 'www.mydom2.com')])
        self.assertStatus('200 OK')
        self.assertHeader('Content-Type', 'text/css;charset=utf-8')
        self.getPage('/static2/dirback.jpg', [('Host', 'www.mydom2.com')])
        self.assertStatus('200 OK')
        self.assertHeaderIn('Content-Type', ['image/jpeg', 'image/pjpeg'])
        self.getPage('/static2/', [('Host', 'www.mydom2.com')])
        self.assertStatus('200 OK')
        self.assertBody('Hello, world\r\n')
        self.getPage('/static2', [('Host', 'www.mydom2.com')])
        self.assertStatus(301)