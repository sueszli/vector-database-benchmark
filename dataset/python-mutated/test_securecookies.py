import unittest
import bottle
from bottle import tob, touni

class TestSignedCookies(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.data = touni('υηι¢σ∂є')
        self.secret = tob('secret')
        bottle.app.push()
        bottle.response.bind()

    def tear_down(self):
        if False:
            return 10
        bottle.app.pop()

    def get_pairs(self):
        if False:
            while True:
                i = 10
        for (k, v) in bottle.response.headerlist:
            if k == 'Set-Cookie':
                (key, value) = v.split(';')[0].split('=', 1)
                yield (key.lower().strip(), value.strip())

    def set_pairs(self, pairs):
        if False:
            i = 10
            return i + 15
        header = ','.join(['%s=%s' % (k, v) for (k, v) in pairs])
        bottle.request.bind({'HTTP_COOKIE': header})

    def testValid(self):
        if False:
            return 10
        bottle.response.set_cookie('key', self.data, secret=self.secret)
        pairs = self.get_pairs()
        self.set_pairs(pairs)
        result = bottle.request.get_cookie('key', secret=self.secret)
        self.assertEqual(self.data, result)

    def testWrongKey(self):
        if False:
            while True:
                i = 10
        bottle.response.set_cookie('key', self.data, secret=self.secret)
        pairs = self.get_pairs()
        self.set_pairs([(k + 'xxx', v) for (k, v) in pairs])
        result = bottle.request.get_cookie('key', secret=self.secret)
        self.assertEqual(None, result)

class TestSignedCookiesWithPickle(TestSignedCookies):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestSignedCookiesWithPickle, self).setUp()
        self.data = dict(a=5, b=touni('υηι¢σ∂є'), c=[1, 2, 3, 4, tob('bytestring')])