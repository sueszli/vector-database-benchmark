import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
localDir = os.path.dirname(__file__)
logfile = os.path.join(localDir, 'test_misc_tools.log')

def setup_server():
    if False:
        while True:
            i = 10

    class Root:

        @cherrypy.expose
        def index(self):
            if False:
                for i in range(10):
                    print('nop')
            yield 'Hello, world'
        h = [('Content-Language', 'en-GB'), ('Content-Type', 'text/plain')]
        tools.response_headers(headers=h)(index)

        @cherrypy.expose
        @cherrypy.config(**{'tools.response_headers.on': True, 'tools.response_headers.headers': [('Content-Language', 'fr'), ('Content-Type', 'text/plain')], 'tools.log_hooks.on': True})
        def other(self):
            if False:
                return 10
            return 'salut'

    @cherrypy.config(**{'tools.accept.on': True})
    class Accept:

        @cherrypy.expose
        def index(self):
            if False:
                return 10
            return '<a href="feed">Atom feed</a>'

        @cherrypy.expose
        @tools.accept(media='application/atom+xml')
        def feed(self):
            if False:
                return 10
            return '<?xml version="1.0" encoding="utf-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">\n    <title>Unknown Blog</title>\n</feed>'

        @cherrypy.expose
        def select(self):
            if False:
                print('Hello World!')
            mtype = tools.accept.callable(['text/html', 'text/plain'])
            if mtype == 'text/html':
                return '<h2>Page Title</h2>'
            else:
                return 'PAGE TITLE'

    class Referer:

        @cherrypy.expose
        def accept(self):
            if False:
                while True:
                    i = 10
            return 'Accepted!'
        reject = accept

    class AutoVary:

        @cherrypy.expose
        def index(self):
            if False:
                i = 10
                return i + 15
            cherrypy.request.headers.get('Accept-Encoding')
            cherrypy.request.headers['Host']
            'If-Modified-Since' in cherrypy.request.headers
            'Range' in cherrypy.request.headers
            tools.accept.callable(['text/html', 'text/plain'])
            return 'Hello, world!'
    conf = {'/referer': {'tools.referer.on': True, 'tools.referer.pattern': 'http://[^/]*example\\.com'}, '/referer/reject': {'tools.referer.accept': False, 'tools.referer.accept_missing': True}, '/autovary': {'tools.autovary.on': True}}
    root = Root()
    root.referer = Referer()
    root.accept = Accept()
    root.autovary = AutoVary()
    cherrypy.tree.mount(root, config=conf)
    cherrypy.config.update({'log.error_file': logfile})

class ResponseHeadersTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def testResponseHeadersDecorator(self):
        if False:
            for i in range(10):
                print('nop')
        self.getPage('/')
        self.assertHeader('Content-Language', 'en-GB')
        self.assertHeader('Content-Type', 'text/plain;charset=utf-8')

    def testResponseHeaders(self):
        if False:
            for i in range(10):
                print('nop')
        self.getPage('/other')
        self.assertHeader('Content-Language', 'fr')
        self.assertHeader('Content-Type', 'text/plain;charset=utf-8')

class RefererTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def testReferer(self):
        if False:
            return 10
        self.getPage('/referer/accept')
        self.assertErrorPage(403, 'Forbidden Referer header.')
        self.getPage('/referer/accept', headers=[('Referer', 'http://www.example.com/')])
        self.assertStatus(200)
        self.assertBody('Accepted!')
        self.getPage('/referer/reject')
        self.assertStatus(200)
        self.assertBody('Accepted!')
        self.getPage('/referer/reject', headers=[('Referer', 'http://www.example.com/')])
        self.assertErrorPage(403, 'Forbidden Referer header.')

class AcceptTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def test_Accept_Tool(self):
        if False:
            print('Hello World!')
        self.getPage('/accept/feed')
        self.assertStatus(200)
        self.assertInBody('<title>Unknown Blog</title>')
        self.getPage('/accept/feed', headers=[('Accept', 'application/atom+xml')])
        self.assertStatus(200)
        self.assertInBody('<title>Unknown Blog</title>')
        self.getPage('/accept/feed', headers=[('Accept', 'application/*')])
        self.assertStatus(200)
        self.assertInBody('<title>Unknown Blog</title>')
        self.getPage('/accept/feed', headers=[('Accept', '*/*')])
        self.assertStatus(200)
        self.assertInBody('<title>Unknown Blog</title>')
        self.getPage('/accept/feed', headers=[('Accept', 'text/html')])
        self.assertErrorPage(406, 'Your client sent this Accept header: text/html. But this resource only emits these media types: application/atom+xml.')
        self.getPage('/accept/')
        self.assertStatus(200)
        self.assertBody('<a href="feed">Atom feed</a>')

    def test_accept_selection(self):
        if False:
            while True:
                i = 10
        self.getPage('/accept/select', [('Accept', 'text/html')])
        self.assertStatus(200)
        self.assertBody('<h2>Page Title</h2>')
        self.getPage('/accept/select', [('Accept', 'text/plain')])
        self.assertStatus(200)
        self.assertBody('PAGE TITLE')
        self.getPage('/accept/select', [('Accept', 'text/plain, text/*;q=0.5')])
        self.assertStatus(200)
        self.assertBody('PAGE TITLE')
        self.getPage('/accept/select', [('Accept', 'text/*')])
        self.assertStatus(200)
        self.assertBody('<h2>Page Title</h2>')
        self.getPage('/accept/select', [('Accept', '*/*')])
        self.assertStatus(200)
        self.assertBody('<h2>Page Title</h2>')
        self.getPage('/accept/select', [('Accept', 'application/xml')])
        self.assertErrorPage(406, 'Your client sent this Accept header: application/xml. But this resource only emits these media types: text/html, text/plain.')

class AutoVaryTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def testAutoVary(self):
        if False:
            return 10
        self.getPage('/autovary/')
        self.assertHeader('Vary', 'Accept, Accept-Charset, Accept-Encoding, Host, If-Modified-Since, Range')