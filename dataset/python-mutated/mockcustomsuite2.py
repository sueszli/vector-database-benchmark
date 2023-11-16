"""
Mock test module that contains a C{testSuite} method. L{runner.TestLoader}
should load the tests from the C{testSuite}, not from the C{Foo} C{TestCase}.

See L{twisted.trial.test.test_loader.LoaderTest.test_loadModuleWith_testSuite}.
"""
from twisted.trial import runner, unittest

class Foo(unittest.SynchronousTestCase):

    def test_foo(self) -> None:
        if False:
            print('Hello World!')
        pass

def testSuite():
    if False:
        while True:
            i = 10
    ts = runner.TestSuite()
    ts.name = 'MyCustomSuite'
    return ts