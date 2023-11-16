import cherrypy
from cherrypy.test import helper

class WSGI_Namespace_Test(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            print('Hello World!')

        class WSGIResponse(object):

            def __init__(self, appresults):
                if False:
                    for i in range(10):
                        print('nop')
                self.appresults = appresults
                self.iter = iter(appresults)

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                return self

            def next(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.iter.next()

            def __next__(self):
                if False:
                    while True:
                        i = 10
                return next(self.iter)

            def close(self):
                if False:
                    for i in range(10):
                        print('nop')
                if hasattr(self.appresults, 'close'):
                    self.appresults.close()

        class ChangeCase(object):

            def __init__(self, app, to=None):
                if False:
                    print('Hello World!')
                self.app = app
                self.to = to

            def __call__(self, environ, start_response):
                if False:
                    print('Hello World!')
                res = self.app(environ, start_response)

                class CaseResults(WSGIResponse):

                    def next(this):
                        if False:
                            for i in range(10):
                                print('nop')
                        return getattr(this.iter.next(), self.to)()

                    def __next__(this):
                        if False:
                            return 10
                        return getattr(next(this.iter), self.to)()
                return CaseResults(res)

        class Replacer(object):

            def __init__(self, app, map={}):
                if False:
                    print('Hello World!')
                self.app = app
                self.map = map

            def __call__(self, environ, start_response):
                if False:
                    print('Hello World!')
                res = self.app(environ, start_response)

                class ReplaceResults(WSGIResponse):

                    def next(this):
                        if False:
                            return 10
                        line = this.iter.next()
                        for (k, v) in self.map.iteritems():
                            line = line.replace(k, v)
                        return line

                    def __next__(this):
                        if False:
                            i = 10
                            return i + 15
                        line = next(this.iter)
                        for (k, v) in self.map.items():
                            line = line.replace(k, v)
                        return line
                return ReplaceResults(res)

        class Root(object):

            @cherrypy.expose
            def index(self):
                if False:
                    print('Hello World!')
                return 'HellO WoRlD!'
        root_conf = {'wsgi.pipeline': [('replace', Replacer)], 'wsgi.replace.map': {b'L': b'X', b'l': b'r'}}
        app = cherrypy.Application(Root())
        app.wsgiapp.pipeline.append(('changecase', ChangeCase))
        app.wsgiapp.config['changecase'] = {'to': 'upper'}
        cherrypy.tree.mount(app, config={'/': root_conf})

    def test_pipeline(self):
        if False:
            while True:
                i = 10
        if not cherrypy.server.httpserver:
            return self.skip()
        self.getPage('/')
        self.assertBody('HERRO WORRD!')