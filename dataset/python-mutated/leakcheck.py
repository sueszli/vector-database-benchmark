from __future__ import print_function
import sys
import gc
from functools import wraps
import unittest
try:
    import objgraph
except ImportError:
    objgraph = None
import gevent
import gevent.core

def ignores_leakcheck(func):
    if False:
        while True:
            i = 10
    '\n    Ignore the given object during leakchecks.\n\n    Can be applied to a method, in which case the method will run, but\n    will not be subject to leak checks.\n\n    If applied to a class, the entire class will be skipped during leakchecks. This\n    is intended to be used for classes that are very slow and cause problems such as\n    test timeouts; typically it will be used for classes that are subclasses of a base\n    class and specify variants of behaviour (such as pool sizes).\n    '
    func.ignore_leakcheck = True
    return func

class _RefCountChecker(object):
    IGNORED_TYPES = (tuple, dict)
    try:
        CALLBACK_KIND = gevent.core.callback
    except AttributeError:
        from gevent._ffi.callback import callback as CALLBACK_KIND

    def __init__(self, testcase, function):
        if False:
            while True:
                i = 10
        self.testcase = testcase
        self.function = function
        self.deltas = []
        self.peak_stats = {}
        self.needs_setUp = False

    def _ignore_object_p(self, obj):
        if False:
            print('Hello World!')
        if obj is self:
            return False
        try:
            if obj in self.__dict__.values() or obj == self._ignore_object_p:
                return False
        except (AttributeError, TypeError):
            return True
        kind = type(obj)
        if kind in self.IGNORED_TYPES:
            return False
        if kind is self.CALLBACK_KIND and obj.callback is None and (obj.args is None):
            return False
        return True

    def _growth(self):
        if False:
            for i in range(10):
                print('nop')
        return objgraph.growth(limit=None, peak_stats=self.peak_stats, filter=self._ignore_object_p)

    def _report_diff(self, growth):
        if False:
            return 10
        if not growth:
            return '<Unable to calculate growth>'
        lines = []
        width = max((len(name) for (name, _, _) in growth))
        for (name, count, delta) in growth:
            lines.append('%-*s%9d %+9d' % (width, name, count, delta))
        diff = '\n'.join(lines)
        return diff

    def _run_test(self, args, kwargs):
        if False:
            print('Hello World!')
        gc_enabled = gc.isenabled()
        gc.disable()
        if self.needs_setUp:
            self.testcase.setUp()
            self.testcase.skipTearDown = False
        try:
            self.function(self.testcase, *args, **kwargs)
        finally:
            self.testcase.tearDown()
            self.testcase.doCleanups()
            self.testcase.skipTearDown = True
            self.needs_setUp = True
            if gc_enabled:
                gc.enable()

    def _growth_after(self):
        if False:
            i = 10
            return i + 15
        if 'urlparse' in sys.modules:
            sys.modules['urlparse'].clear_cache()
        if 'urllib.parse' in sys.modules:
            sys.modules['urllib.parse'].clear_cache()
        return self._growth()

    def _check_deltas(self, growth):
        if False:
            for i in range(10):
                print('nop')
        deltas = self.deltas
        if not deltas:
            return True
        if gc.garbage:
            raise AssertionError('Generated uncollectable garbage %r' % (gc.garbage,))
        if deltas[-2:] == [0, 0] and len(deltas) in (2, 3):
            return False
        if deltas[-3:] == [0, 0, 0]:
            return False
        if len(deltas) >= 4 and sum(deltas[-4:]) == 0:
            return False
        if len(deltas) >= 3 and deltas[-1] > 0 and (deltas[-1] == deltas[-2]) and (deltas[-2] == deltas[-3]):
            diff = self._report_diff(growth)
            raise AssertionError('refcount increased by %r\n%s' % (deltas, diff))
        if sum(deltas[-3:]) <= 0 or sum(deltas[-4:]) <= 0 or deltas[-4:].count(0) >= 2:
            limit = 11
        else:
            limit = 7
        if len(deltas) >= limit:
            raise AssertionError('refcount increased by %r\n%s' % (deltas, self._report_diff(growth)))
        return True

    def __call__(self, args, kwargs):
        if False:
            print('Hello World!')
        for _ in range(3):
            gc.collect()
        growth = self._growth()
        while self._check_deltas(growth):
            self._run_test(args, kwargs)
            growth = self._growth_after()
            self.deltas.append(sum((stat[2] for stat in growth)))

def wrap_refcount(method):
    if False:
        for i in range(10):
            print('nop')
    if objgraph is None or getattr(method, 'ignore_leakcheck', False):
        if objgraph is None:
            import warnings
            warnings.warn('objgraph not available, leakchecks disabled')

        @wraps(method)
        def _method_skipped_during_leakcheck(self, *_args, **_kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self.skipTest('This method ignored during leakchecks')
        return _method_skipped_during_leakcheck

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if getattr(self, 'ignore_leakcheck', False):
            raise unittest.SkipTest('This class ignored during leakchecks')
        return _RefCountChecker(self, method)(args, kwargs)
    return wrapper