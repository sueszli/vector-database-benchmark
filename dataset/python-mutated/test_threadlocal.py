import unittest
from pyramid import testing

class TestThreadLocalManager(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        testing.setUp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        testing.tearDown()

    def _getTargetClass(self):
        if False:
            return 10
        from pyramid.threadlocal import ThreadLocalManager
        return ThreadLocalManager

    def _makeOne(self, default=lambda *x: 1):
        if False:
            i = 10
            return i + 15
        return self._getTargetClass()(default)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        local = self._makeOne()
        self.assertEqual(local.stack, [])
        self.assertEqual(local.get(), 1)

    def test_default(self):
        if False:
            while True:
                i = 10

        def thedefault():
            if False:
                i = 10
                return i + 15
            return '123'
        local = self._makeOne(thedefault)
        self.assertEqual(local.stack, [])
        self.assertEqual(local.get(), '123')

    def test_push_and_pop(self):
        if False:
            i = 10
            return i + 15
        local = self._makeOne()
        local.push(True)
        self.assertEqual(local.get(), True)
        self.assertEqual(local.pop(), True)
        self.assertEqual(local.pop(), None)
        self.assertEqual(local.get(), 1)

    def test_set_get_and_clear(self):
        if False:
            i = 10
            return i + 15
        local = self._makeOne()
        local.set(None)
        self.assertEqual(local.stack, [None])
        self.assertEqual(local.get(), None)
        local.clear()
        self.assertEqual(local.get(), 1)
        local.clear()
        self.assertEqual(local.get(), 1)

class TestGetCurrentRequest(unittest.TestCase):

    def _callFUT(self):
        if False:
            i = 10
            return i + 15
        from pyramid.threadlocal import get_current_request
        return get_current_request()

    def test_it_None(self):
        if False:
            for i in range(10):
                print('nop')
        request = self._callFUT()
        self.assertEqual(request, None)

    def test_it(self):
        if False:
            while True:
                i = 10
        from pyramid.threadlocal import manager
        request = object()
        try:
            manager.push({'request': request})
            self.assertEqual(self._callFUT(), request)
        finally:
            manager.pop()
        self.assertEqual(self._callFUT(), None)

class GetCurrentRegistryTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        testing.setUp()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        testing.tearDown()

    def _callFUT(self):
        if False:
            while True:
                i = 10
        from pyramid.threadlocal import get_current_registry
        return get_current_registry()

    def test_it(self):
        if False:
            return 10
        from pyramid.threadlocal import manager
        try:
            manager.push({'registry': 123})
            self.assertEqual(self._callFUT(), 123)
        finally:
            manager.pop()

class GetCurrentRegistryWithoutTestingRegistry(unittest.TestCase):

    def _callFUT(self):
        if False:
            i = 10
            return i + 15
        from pyramid.threadlocal import get_current_registry
        return get_current_registry()

    def test_it(self):
        if False:
            while True:
                i = 10
        from pyramid.registry import global_registry
        self.assertEqual(self._callFUT(), global_registry)