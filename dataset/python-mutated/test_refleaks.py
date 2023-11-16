"""Tests for refleaks."""
import itertools
import platform
import threading
from http.client import HTTPConnection
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.test import helper
data = object()

class ReferenceTests(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            i = 10
            return i + 15

        class Root:

            @cherrypy.expose
            def index(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                cherrypy.request.thing = data
                return 'Hello world!'
        cherrypy.tree.mount(Root())

    def test_threadlocal_garbage(self):
        if False:
            return 10
        if platform.system() == 'Darwin':
            self.skip('queue issues; see #1474')
        success = itertools.count()

        def getpage():
            if False:
                for i in range(10):
                    print('nop')
            host = '%s:%s' % (self.interface(), self.PORT)
            if self.scheme == 'https':
                c = HTTPSConnection(host)
            else:
                c = HTTPConnection(host)
            try:
                c.putrequest('GET', '/')
                c.endheaders()
                response = c.getresponse()
                body = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(body, b'Hello world!')
            finally:
                c.close()
            next(success)
        ITERATIONS = 25
        ts = [threading.Thread(target=getpage) for _ in range(ITERATIONS)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(next(success), ITERATIONS)