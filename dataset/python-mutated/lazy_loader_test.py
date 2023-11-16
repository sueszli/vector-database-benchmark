"""lazy loader tests."""
import doctest
import inspect
import types
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import tf_inspect

class LazyLoaderTest(test.TestCase):

    def testDocTestDoesNotLoad(self):
        if False:
            while True:
                i = 10
        module = types.ModuleType('mytestmodule')
        module.foo = lazy_loader.LazyLoader('foo', module.__dict__, 'os.path')
        self.assertIsInstance(module.foo, lazy_loader.LazyLoader)
        finder = doctest.DocTestFinder()
        finder.find(module)
        self.assertIsInstance(module.foo, lazy_loader.LazyLoader)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def testLazyLoaderMock(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        name = LazyLoaderTest.__module__
        lazy_loader_module = lazy_loader.LazyLoader('lazy_loader_module', globals(), name, warning='Test warning.')
        self.assertEqual(0, mock_warning.call_count)
        lazy_loader_module.foo = 0
        self.assertEqual(1, mock_warning.call_count)
        foo = lazy_loader_module.foo
        self.assertEqual(1, mock_warning.call_count)
        self.assertEqual(lazy_loader_module.foo, foo)
if __name__ == '__main__':
    test.main()