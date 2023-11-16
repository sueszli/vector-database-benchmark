from twisted.trial import unittest

class Foo(unittest.SynchronousTestCase):

    def testFoo(self) -> None:
        if False:
            print('Hello World!')
        pass