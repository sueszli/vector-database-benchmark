import unittest
from twisted.internet import defer

class TestBleeding(unittest.TestCase):
    """This test creates an unhandled Deferred and leaves it in a cycle.

    The Deferred is left in a cycle so that the garbage collector won't pick it
    up immediately.  We were having some problems where unhandled Deferreds in
    one test were failing random other tests. (See #1507, #1213)
    """

    def test_unhandledDeferred(self):
        if False:
            while True:
                i = 10
        try:
            1 / 0
        except ZeroDivisionError:
            f = defer.fail()
        l = [f]
        l.append(l)