import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.test import helper

class ETagTest(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            return 10

        class Root:

            @cherrypy.expose
            def resource(self):
                if False:
                    i = 10
                    return i + 15
                return 'Oh wah ta goo Siam.'

            @cherrypy.expose
            def fail(self, code):
                if False:
                    i = 10
                    return i + 15
                code = int(code)
                if 300 <= code <= 399:
                    raise cherrypy.HTTPRedirect([], code)
                else:
                    raise cherrypy.HTTPError(code)

            @cherrypy.expose
            @cherrypy.config(**{'tools.encode.on': True})
            def unicoded(self):
                if False:
                    return 10
                return ntou('I am a Ụnicode string.', 'escape')
        conf = {'/': {'tools.etags.on': True, 'tools.etags.autotags': True}}
        cherrypy.tree.mount(Root(), config=conf)

    def test_etags(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/resource')
        self.assertStatus('200 OK')
        self.assertHeader('Content-Type', 'text/html;charset=utf-8')
        self.assertBody('Oh wah ta goo Siam.')
        etag = self.assertHeader('ETag')
        self.getPage('/resource', headers=[('If-Match', etag)])
        self.assertStatus('200 OK')
        self.getPage('/resource', headers=[('If-Match', '*')])
        self.assertStatus('200 OK')
        self.getPage('/resource', headers=[('If-Match', '*')], method='POST')
        self.assertStatus('200 OK')
        self.getPage('/resource', headers=[('If-Match', 'a bogus tag')])
        self.assertStatus('412 Precondition Failed')
        self.getPage('/resource', headers=[('If-None-Match', etag)])
        self.assertStatus(304)
        self.getPage('/resource', method='POST', headers=[('If-None-Match', etag)])
        self.assertStatus('412 Precondition Failed')
        self.getPage('/resource', headers=[('If-None-Match', '*')])
        self.assertStatus(304)
        self.getPage('/resource', headers=[('If-None-Match', 'a bogus tag')])
        self.assertStatus('200 OK')

    def test_errors(self):
        if False:
            while True:
                i = 10
        self.getPage('/resource')
        self.assertStatus(200)
        etag = self.assertHeader('ETag')
        self.getPage('/fail/412', headers=[('If-Match', etag)])
        self.assertStatus(412)
        self.getPage('/fail/304', headers=[('If-Match', etag)])
        self.assertStatus(304)
        self.getPage('/fail/412', headers=[('If-None-Match', '*')])
        self.assertStatus(412)
        self.getPage('/fail/304', headers=[('If-None-Match', '*')])
        self.assertStatus(304)

    def test_unicode_body(self):
        if False:
            print('Hello World!')
        self.getPage('/unicoded')
        self.assertStatus(200)
        etag1 = self.assertHeader('ETag')
        self.getPage('/unicoded', headers=[('If-Match', etag1)])
        self.assertStatus(200)
        self.assertHeader('ETag', etag1)