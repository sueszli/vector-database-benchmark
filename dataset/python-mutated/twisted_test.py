import unittest
from tornado.testing import AsyncTestCase, gen_test
try:
    from twisted.internet.defer import inlineCallbacks, returnValue
    have_twisted = True
except ImportError:
    have_twisted = False
else:
    import tornado.platform.twisted
skipIfNoTwisted = unittest.skipUnless(have_twisted, 'twisted module not present')

@skipIfNoTwisted
class ConvertDeferredTest(AsyncTestCase):

    @gen_test
    def test_success(self):
        if False:
            i = 10
            return i + 15

        @inlineCallbacks
        def fn():
            if False:
                while True:
                    i = 10
            if False:
                yield
            returnValue(42)
        res = (yield fn())
        self.assertEqual(res, 42)

    @gen_test
    def test_failure(self):
        if False:
            while True:
                i = 10

        @inlineCallbacks
        def fn():
            if False:
                return 10
            if False:
                yield
            1 / 0
        with self.assertRaises(ZeroDivisionError):
            yield fn()
if __name__ == '__main__':
    unittest.main()