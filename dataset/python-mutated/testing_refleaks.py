"""A subclass of unittest.TestCase which checks for reference leaks.

To use:
- Use testing_refleak.BaseTestCase instead of unittest.TestCase
- Configure and compile Python with --with-pydebug

If sys.gettotalrefcount() is not available (because Python was built without
the Py_DEBUG option), then this module is a no-op and tests will run normally.
"""
import gc
import sys
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg
try:
    import unittest2 as unittest
except ImportError:
    import unittest

class LocalTestResult(unittest.TestResult):
    """A TestResult which forwards events to a parent object, except for Skips."""

    def __init__(self, parent_result):
        if False:
            for i in range(10):
                print('nop')
        unittest.TestResult.__init__(self)
        self.parent_result = parent_result

    def addError(self, test, error):
        if False:
            i = 10
            return i + 15
        self.parent_result.addError(test, error)

    def addFailure(self, test, error):
        if False:
            i = 10
            return i + 15
        self.parent_result.addFailure(test, error)

    def addSkip(self, test, reason):
        if False:
            print('Hello World!')
        pass

class ReferenceLeakCheckerTestCase(unittest.TestCase):
    """A TestCase which runs tests multiple times, collecting reference counts."""
    NB_RUNS = 3

    def run(self, result=None):
        if False:
            while True:
                i = 10
        self._saved_pickle_registry = copyreg.dispatch_table.copy()
        super(ReferenceLeakCheckerTestCase, self).run(result=result)
        super(ReferenceLeakCheckerTestCase, self).run(result=result)
        oldrefcount = 0
        local_result = LocalTestResult(result)
        refcount_deltas = []
        for _ in range(self.NB_RUNS):
            oldrefcount = self._getRefcounts()
            super(ReferenceLeakCheckerTestCase, self).run(result=local_result)
            newrefcount = self._getRefcounts()
            refcount_deltas.append(newrefcount - oldrefcount)
        print(refcount_deltas, self)
        try:
            self.assertEqual(refcount_deltas, [0] * self.NB_RUNS)
        except Exception:
            result.addError(self, sys.exc_info())

    def _getRefcounts(self):
        if False:
            while True:
                i = 10
        copyreg.dispatch_table.clear()
        copyreg.dispatch_table.update(self._saved_pickle_registry)
        gc.collect()
        gc.collect()
        gc.collect()
        return sys.gettotalrefcount()
if hasattr(sys, 'gettotalrefcount'):
    BaseTestCase = ReferenceLeakCheckerTestCase
    SkipReferenceLeakChecker = unittest.skip
else:
    BaseTestCase = unittest.TestCase

    def SkipReferenceLeakChecker(reason):
        if False:
            for i in range(10):
                print('nop')
        del reason

        def Same(func):
            if False:
                while True:
                    i = 10
            return func
        return Same