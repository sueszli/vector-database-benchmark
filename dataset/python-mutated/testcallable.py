import unittest
from unittest.test.testmock.support import is_instance, X, SomeClass
from unittest.mock import Mock, MagicMock, NonCallableMagicMock, NonCallableMock, patch, create_autospec, CallableMixin

class TestCallable(unittest.TestCase):

    def assertNotCallable(self, mock):
        if False:
            return 10
        self.assertTrue(is_instance(mock, NonCallableMagicMock))
        self.assertFalse(is_instance(mock, CallableMixin))

    def test_non_callable(self):
        if False:
            i = 10
            return i + 15
        for mock in (NonCallableMagicMock(), NonCallableMock()):
            self.assertRaises(TypeError, mock)
            self.assertFalse(hasattr(mock, '__call__'))
            self.assertIn(mock.__class__.__name__, repr(mock))

    def test_hierarchy(self):
        if False:
            while True:
                i = 10
        self.assertTrue(issubclass(MagicMock, Mock))
        self.assertTrue(issubclass(NonCallableMagicMock, NonCallableMock))

    def test_attributes(self):
        if False:
            while True:
                i = 10
        one = NonCallableMock()
        self.assertTrue(issubclass(type(one.one), Mock))
        two = NonCallableMagicMock()
        self.assertTrue(issubclass(type(two.two), MagicMock))

    def test_subclasses(self):
        if False:
            for i in range(10):
                print('nop')

        class MockSub(Mock):
            pass
        one = MockSub()
        self.assertTrue(issubclass(type(one.one), MockSub))

        class MagicSub(MagicMock):
            pass
        two = MagicSub()
        self.assertTrue(issubclass(type(two.two), MagicSub))

    def test_patch_spec(self):
        if False:
            print('Hello World!')
        patcher = patch('%s.X' % __name__, spec=True)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        instance = mock()
        mock.assert_called_once_with()
        self.assertNotCallable(instance)
        self.assertRaises(TypeError, instance)

    def test_patch_spec_set(self):
        if False:
            print('Hello World!')
        patcher = patch('%s.X' % __name__, spec_set=True)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        instance = mock()
        mock.assert_called_once_with()
        self.assertNotCallable(instance)
        self.assertRaises(TypeError, instance)

    def test_patch_spec_instance(self):
        if False:
            for i in range(10):
                print('nop')
        patcher = patch('%s.X' % __name__, spec=X())
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        self.assertNotCallable(mock)
        self.assertRaises(TypeError, mock)

    def test_patch_spec_set_instance(self):
        if False:
            i = 10
            return i + 15
        patcher = patch('%s.X' % __name__, spec_set=X())
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        self.assertNotCallable(mock)
        self.assertRaises(TypeError, mock)

    def test_patch_spec_callable_class(self):
        if False:
            i = 10
            return i + 15

        class CallableX(X):

            def __call__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class Sub(CallableX):
            pass

        class Multi(SomeClass, Sub):
            pass
        for arg in ('spec', 'spec_set'):
            for Klass in (CallableX, Sub, Multi):
                with patch('%s.X' % __name__, **{arg: Klass}) as mock:
                    instance = mock()
                    mock.assert_called_once_with()
                    self.assertTrue(is_instance(instance, MagicMock))
                    self.assertRaises(AttributeError, getattr, instance, 'foobarbaz')
                    result = instance()
                    instance.assert_called_once_with()
                    result(3, 2, 1)
                    result.assert_called_once_with(3, 2, 1)
                    result.foo(3, 2, 1)
                    result.foo.assert_called_once_with(3, 2, 1)

    def test_create_autospec(self):
        if False:
            print('Hello World!')
        mock = create_autospec(X)
        instance = mock()
        self.assertRaises(TypeError, instance)
        mock = create_autospec(X())
        self.assertRaises(TypeError, mock)

    def test_create_autospec_instance(self):
        if False:
            while True:
                i = 10
        mock = create_autospec(SomeClass, instance=True)
        self.assertRaises(TypeError, mock)
        mock.wibble()
        mock.wibble.assert_called_once_with()
        self.assertRaises(TypeError, mock.wibble, 'some', 'args')
if __name__ == '__main__':
    unittest.main()