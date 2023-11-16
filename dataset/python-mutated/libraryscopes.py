import inspect
from robot.utils import normalize

def LibraryScope(libcode, library):
    if False:
        i = 10
        return i + 15
    scope = _get_scope(libcode)
    if scope == 'GLOBAL':
        return GlobalScope(library)
    if scope in ('SUITE', 'TESTSUITE'):
        return TestSuiteScope(library)
    return TestCaseScope(library)

def _get_scope(libcode):
    if False:
        for i in range(10):
            print('nop')
    if inspect.ismodule(libcode):
        return 'GLOBAL'
    scope = getattr(libcode, 'ROBOT_LIBRARY_SCOPE', '')
    return normalize(str(scope), ignore='_').upper()

class GlobalScope:
    is_global = True

    def __init__(self, library):
        if False:
            print('Hello World!')
        self._register_listeners = library.register_listeners
        self._unregister_listeners = library.unregister_listeners

    def start_suite(self):
        if False:
            for i in range(10):
                print('nop')
        self._register_listeners()

    def end_suite(self):
        if False:
            while True:
                i = 10
        self._unregister_listeners()

    def start_test(self):
        if False:
            i = 10
            return i + 15
        pass

    def end_test(self):
        if False:
            return 10
        pass

    def __str__(self):
        if False:
            print('Hello World!')
        return 'GLOBAL'

class TestSuiteScope(GlobalScope):
    is_global = False

    def __init__(self, library):
        if False:
            print('Hello World!')
        GlobalScope.__init__(self, library)
        self._reset_instance = library.reset_instance
        self._instance_cache = []

    def start_suite(self):
        if False:
            return 10
        prev = self._reset_instance()
        self._instance_cache.append(prev)
        self._register_listeners()

    def end_suite(self):
        if False:
            while True:
                i = 10
        self._unregister_listeners(close=True)
        prev = self._instance_cache.pop()
        self._reset_instance(prev)

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'SUITE'

class TestCaseScope(TestSuiteScope):

    def start_test(self):
        if False:
            while True:
                i = 10
        self._unregister_listeners()
        prev = self._reset_instance()
        self._instance_cache.append(prev)
        self._register_listeners()

    def end_test(self):
        if False:
            return 10
        self._unregister_listeners(close=True)
        prev = self._instance_cache.pop()
        self._reset_instance(prev)
        self._register_listeners()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'TEST'