"""
Mock test module that contains a C{test_suite} method. L{runner.TestLoader}
should load the tests from the C{test_suite}, not from the C{Foo} C{TestCase}.

See {twisted.trial.test.test_loader.LoaderTest.test_loadModuleWith_test_suite}.
"""
from twisted.trial import runner, unittest

class Foo(unittest.SynchronousTestCase):

    def test_foo(self) -> None:
        if False:
            while True:
                i = 10
        pass

def test_suite():
    if False:
        return 10
    ts = runner.TestSuite()
    ts.name = 'MyCustomSuite'
    return ts