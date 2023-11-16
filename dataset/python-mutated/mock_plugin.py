from ckan.plugins import Plugin, SingletonPlugin

class _MockPlugin(object):
    """
    MockPlugin tracks method calls via __getattr__ for rapid mocking of
    plugins.

    Use MockPlugin.calls or MockPlugin.<methodname>.calls to access
    call information
    """

    class MockMethod(object):
        registry = {}

        def __init__(self, boundto, name):
            if False:
                print('Hello World!')
            self.name = name
            self.calls = []
            self.boundto = boundto

        def __call__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            self.boundto.calls.append((self.name, args, kwargs))
            self.calls.append((args, kwargs))

    def __init__(self, *arg, **kw):
        if False:
            i = 10
            return i + 15
        self.calls = []
        self.__mockmethods__ = {}

    def __getattr__(self, name):
        if False:
            return 10
        if name not in self.__mockmethods__:
            self.__mockmethods__[name] = self.MockMethod(self, name)
        return self.__mockmethods__[name]

    def reset_calls(self):
        if False:
            i = 10
            return i + 15
        '\n        Reset call information for this instance\n        '
        for mockmethod in self.MockMethod.registry.values():
            mockmethod.calls = []
        self.__mockmethods__ = {}
        self.calls = []

class MockPlugin(_MockPlugin, Plugin):
    """
    Mock a plugin
    """

class MockSingletonPlugin(_MockPlugin, SingletonPlugin):
    """
    Mock a singleton plugin
    """