from collections import deque
import unittest

class Test_get_root(unittest.TestCase):

    def _callFUT(self, app, request=None):
        if False:
            for i in range(10):
                print('nop')
        from pyramid.scripting import get_root
        return get_root(app, request)

    def _makeRegistry(self):
        if False:
            return 10
        return DummyRegistry([DummyFactory])

    def setUp(self):
        if False:
            i = 10
            return i + 15
        from pyramid.threadlocal import manager
        self.manager = manager
        self.default = manager.get()

    def test_it_norequest(self):
        if False:
            i = 10
            return i + 15
        registry = self._makeRegistry()
        app = DummyApp(registry=registry)
        (root, closer) = self._callFUT(app)
        self.assertEqual(dummy_root, root)
        pushed = self.manager.get()
        self.assertEqual(pushed['registry'], registry)
        self.assertEqual(pushed['request'].registry, registry)
        self.assertEqual(pushed['request'].environ['path'], '/')
        closer()
        self.assertEqual(self.default, self.manager.get())

    def test_it_withrequest(self):
        if False:
            while True:
                i = 10
        registry = self._makeRegistry()
        app = DummyApp(registry=registry)
        request = DummyRequest({})
        (root, closer) = self._callFUT(app, request)
        self.assertEqual(dummy_root, root)
        pushed = self.manager.get()
        self.assertEqual(pushed['registry'], registry)
        self.assertEqual(pushed['request'], request)
        self.assertEqual(pushed['request'].registry, registry)
        closer()
        self.assertEqual(self.default, self.manager.get())

class Test_prepare(unittest.TestCase):

    def _callFUT(self, request=None, registry=None):
        if False:
            print('Hello World!')
        from pyramid.scripting import prepare
        return prepare(request, registry)

    def _makeRegistry(self, L=None):
        if False:
            i = 10
            return i + 15
        if L is None:
            L = [None, DummyFactory]
        return DummyRegistry(L)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        from pyramid.threadlocal import manager
        self.manager = manager
        self.default = manager.get()

    def test_it_no_valid_apps(self):
        if False:
            while True:
                i = 10
        from pyramid.exceptions import ConfigurationError
        self.assertRaises(ConfigurationError, self._callFUT)

    def test_it_norequest(self):
        if False:
            print('Hello World!')
        registry = self._makeRegistry([DummyFactory, None, DummyFactory])
        info = self._callFUT(registry=registry)
        (root, closer, request) = (info['root'], info['closer'], info['request'])
        pushed = self.manager.get()
        self.assertEqual(pushed['registry'], registry)
        self.assertEqual(pushed['request'].registry, registry)
        self.assertEqual(root.a, (pushed['request'],))
        closer()
        self.assertEqual(self.default, self.manager.get())
        self.assertEqual(request.context, root)

    def test_it_withrequest_hasregistry(self):
        if False:
            for i in range(10):
                print('nop')
        request = DummyRequest({})
        registry = request.registry = self._makeRegistry()
        info = self._callFUT(request=request)
        (root, closer, request) = (info['root'], info['closer'], info['request'])
        pushed = self.manager.get()
        self.assertEqual(pushed['request'], request)
        self.assertEqual(pushed['registry'], registry)
        self.assertEqual(pushed['request'].registry, registry)
        self.assertEqual(root.a, (request,))
        closer()
        self.assertEqual(self.default, self.manager.get())
        self.assertEqual(request.context, root)
        self.assertEqual(request.registry, registry)

    def test_it_withrequest_noregistry(self):
        if False:
            i = 10
            return i + 15
        request = DummyRequest({})
        registry = self._makeRegistry()
        info = self._callFUT(request=request, registry=registry)
        (root, closer, request) = (info['root'], info['closer'], info['request'])
        closer()
        self.assertEqual(request.context, root)
        self.assertEqual(request.registry, registry)

    def test_it_with_request_and_registry(self):
        if False:
            return 10
        request = DummyRequest({})
        registry = request.registry = self._makeRegistry()
        info = self._callFUT(request=request, registry=registry)
        (root, closer, root) = (info['root'], info['closer'], info['root'])
        pushed = self.manager.get()
        self.assertEqual(pushed['request'], request)
        self.assertEqual(pushed['registry'], registry)
        self.assertEqual(pushed['request'].registry, registry)
        self.assertEqual(root.a, (request,))
        closer()
        self.assertEqual(self.default, self.manager.get())
        self.assertEqual(request.context, root)

    def test_it_with_request_context_already_set(self):
        if False:
            while True:
                i = 10
        request = DummyRequest({})
        context = Dummy()
        request.context = context
        registry = request.registry = self._makeRegistry()
        info = self._callFUT(request=request, registry=registry)
        closer = info['closer']
        closer()
        self.assertEqual(request.context, context)

    def test_it_with_extensions(self):
        if False:
            while True:
                i = 10
        from pyramid.util import InstancePropertyHelper
        exts = DummyExtensions()
        ext_method = lambda r: 'bar'
        (name, fn) = InstancePropertyHelper.make_property(ext_method, 'foo')
        exts.descriptors[name] = fn
        request = DummyRequest({})
        registry = request.registry = self._makeRegistry([exts, DummyFactory])
        info = self._callFUT(request=request, registry=registry)
        self.assertEqual(request.foo, 'bar')
        closer = info['closer']
        closer()

    def test_it_is_a_context_manager(self):
        if False:
            while True:
                i = 10
        request = DummyRequest({})
        registry = request.registry = self._makeRegistry()
        closer_called = [False]
        with self._callFUT(request=request) as info:
            (root, request) = (info['root'], info['request'])
            pushed = self.manager.get()
            self.assertEqual(pushed['request'], request)
            self.assertEqual(pushed['registry'], registry)
            self.assertEqual(pushed['request'].registry, registry)
            self.assertEqual(root.a, (request,))
            orig_closer = info['closer']

            def closer():
                if False:
                    for i in range(10):
                        print('nop')
                orig_closer()
                closer_called[0] = True
            info['closer'] = closer
        self.assertTrue(closer_called[0])
        self.assertEqual(self.default, self.manager.get())
        self.assertEqual(request.context, root)
        self.assertEqual(request.registry, registry)

    def test_closer_invokes_finished_callbacks(self):
        if False:
            i = 10
            return i + 15
        finish_called = [False]

        def finished_callback(request):
            if False:
                i = 10
                return i + 15
            finish_called[0] = True
        request = DummyRequest({})
        request.registry = self._makeRegistry()
        info = self._callFUT(request=request)
        request.add_finished_callback(finished_callback)
        closer = info['closer']
        closer()
        self.assertTrue(finish_called[0])

class Test__make_request(unittest.TestCase):

    def _callFUT(self, path='/', registry=None):
        if False:
            for i in range(10):
                print('nop')
        from pyramid.scripting import _make_request
        return _make_request(path, registry)

    def _makeRegistry(self):
        if False:
            while True:
                i = 10
        return DummyRegistry([DummyFactory])

    def test_it_with_registry(self):
        if False:
            for i in range(10):
                print('nop')
        registry = self._makeRegistry()
        request = self._callFUT('/', registry)
        self.assertEqual(request.environ['path'], '/')
        self.assertEqual(request.registry, registry)

    def test_it_with_no_registry(self):
        if False:
            while True:
                i = 10
        from pyramid.config import global_registries
        registry = self._makeRegistry()
        global_registries.add(registry)
        try:
            request = self._callFUT('/hello')
            self.assertEqual(request.environ['path'], '/hello')
            self.assertEqual(request.registry, registry)
        finally:
            global_registries.empty()

class Dummy:
    pass
dummy_root = Dummy()

class DummyFactory:

    @classmethod
    def blank(cls, path):
        if False:
            print('Hello World!')
        req = DummyRequest({'path': path})
        return req

    def __init__(self, *a, **kw):
        if False:
            while True:
                i = 10
        self.a = a
        self.kw = kw

class DummyRegistry:

    def __init__(self, utilities):
        if False:
            print('Hello World!')
        self.utilities = utilities

    def queryUtility(self, iface, default=None):
        if False:
            print('Hello World!')
        if self.utilities:
            return self.utilities.pop(0)
        return default

class DummyApp:

    def __init__(self, registry=None):
        if False:
            return 10
        if registry:
            self.registry = registry

    def root_factory(self, environ):
        if False:
            for i in range(10):
                print('nop')
        return dummy_root

class DummyRequest:
    matchdict = None
    matched_route = None

    def __init__(self, environ):
        if False:
            return 10
        self.environ = environ
        self.finished_callbacks = deque()

    def add_finished_callback(self, cb):
        if False:
            return 10
        self.finished_callbacks.append(cb)

    def _process_finished_callbacks(self):
        if False:
            for i in range(10):
                print('nop')
        while self.finished_callbacks:
            cb = self.finished_callbacks.popleft()
            cb(self)

class DummyExtensions:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.descriptors = {}
        self.methods = {}