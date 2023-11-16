"""
Test cases used to make sure that warning suppression works at the module,
method, and class levels.

See the L{twisted.trial.test.test_tests} module docstring for details about how
this code is arranged.
"""
import warnings
from twisted.trial import unittest, util
METHOD_WARNING_MSG = 'method warning message'
CLASS_WARNING_MSG = 'class warning message'
MODULE_WARNING_MSG = 'module warning message'

class MethodWarning(Warning):
    pass

class ClassWarning(Warning):
    pass

class ModuleWarning(Warning):
    pass

class EmitMixin:
    """
    Mixin for emiting a variety of warnings.
    """

    def _emit(self):
        if False:
            return 10
        warnings.warn(METHOD_WARNING_MSG, MethodWarning)
        warnings.warn(CLASS_WARNING_MSG, ClassWarning)
        warnings.warn(MODULE_WARNING_MSG, ModuleWarning)

class SuppressionMixin(EmitMixin):
    suppress = [util.suppress(message=CLASS_WARNING_MSG)]

    def testSuppressMethod(self):
        if False:
            return 10
        self._emit()
    testSuppressMethod.suppress = [util.suppress(message=METHOD_WARNING_MSG)]

    def testSuppressClass(self):
        if False:
            print('Hello World!')
        self._emit()

    def testOverrideSuppressClass(self):
        if False:
            return 10
        self._emit()
    testOverrideSuppressClass.suppress = []

class SetUpSuppressionMixin:

    def setUp(self):
        if False:
            while True:
                i = 10
        self._emit()

class TearDownSuppressionMixin:

    def tearDown(self):
        if False:
            print('Hello World!')
        self._emit()

class TestSuppression2Mixin(EmitMixin):

    def testSuppressModule(self):
        if False:
            i = 10
            return i + 15
        self._emit()
suppress = [util.suppress(message=MODULE_WARNING_MSG)]

class SynchronousTestSuppression(SuppressionMixin, unittest.SynchronousTestCase):
    pass

class SynchronousTestSetUpSuppression(SetUpSuppressionMixin, SynchronousTestSuppression):
    pass

class SynchronousTestTearDownSuppression(TearDownSuppressionMixin, SynchronousTestSuppression):
    pass

class SynchronousTestSuppression2(TestSuppression2Mixin, unittest.SynchronousTestCase):
    pass

class AsynchronousTestSuppression(SuppressionMixin, unittest.TestCase):
    pass

class AsynchronousTestSetUpSuppression(SetUpSuppressionMixin, AsynchronousTestSuppression):
    pass

class AsynchronousTestTearDownSuppression(TearDownSuppressionMixin, AsynchronousTestSuppression):
    pass

class AsynchronousTestSuppression2(TestSuppression2Mixin, unittest.TestCase):
    pass