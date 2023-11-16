"""
Tests for warning suppression features of Trial.
"""
import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression

class SuppressionMixin:
    """
    Tests for the warning suppression features of
    L{twisted.trial.unittest.SynchronousTestCase}.
    """

    def runTests(self, suite):
        if False:
            i = 10
            return i + 15
        suite.run(pyunit.TestResult())

    def _load(self, cls, methodName):
        if False:
            while True:
                i = 10
        '\n        Return a new L{unittest.TestSuite} with a single test method in it.\n\n        @param cls: A L{TestCase} subclass defining a test method.\n\n        @param methodName: The name of the test method from C{cls}.\n        '
        return pyunit.TestSuite([cls(methodName)])

    def _assertWarnings(self, warnings, which):
        if False:
            return 10
        '\n        Assert that a certain number of warnings with certain messages were\n        emitted in a certain order.\n\n        @param warnings: A list of emitted warnings, as returned by\n            C{flushWarnings}.\n\n        @param which: A list of strings giving warning messages that should\n            appear in C{warnings}.\n\n        @raise self.failureException: If the warning messages given by C{which}\n            do not match the messages in the warning information in C{warnings},\n            or if they do not appear in the same order.\n        '
        self.assertEqual([warning['message'] for warning in warnings], which)

    def test_setUpSuppression(self):
        if False:
            return 10
        '\n        Suppressions defined by the test method being run are applied to any\n        warnings emitted while running the C{setUp} fixture.\n        '
        self.runTests(self._load(self.TestSetUpSuppression, 'testSuppressMethod'))
        warningsShown = self.flushWarnings([self.TestSetUpSuppression._emit])
        self._assertWarnings(warningsShown, [suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG, suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG])

    def test_tearDownSuppression(self):
        if False:
            return 10
        '\n        Suppressions defined by the test method being run are applied to any\n        warnings emitted while running the C{tearDown} fixture.\n        '
        self.runTests(self._load(self.TestTearDownSuppression, 'testSuppressMethod'))
        warningsShown = self.flushWarnings([self.TestTearDownSuppression._emit])
        self._assertWarnings(warningsShown, [suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG, suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG])

    def test_suppressMethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A suppression set on a test method prevents warnings emitted by that\n        test method which the suppression matches from being emitted.\n        '
        self.runTests(self._load(self.TestSuppression, 'testSuppressMethod'))
        warningsShown = self.flushWarnings([self.TestSuppression._emit])
        self._assertWarnings(warningsShown, [suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG])

    def test_suppressClass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A suppression set on a L{SynchronousTestCase} subclass prevents warnings\n        emitted by any test methods defined on that class which match the\n        suppression from being emitted.\n        '
        self.runTests(self._load(self.TestSuppression, 'testSuppressClass'))
        warningsShown = self.flushWarnings([self.TestSuppression._emit])
        self.assertEqual(warningsShown[0]['message'], suppression.METHOD_WARNING_MSG)
        self.assertEqual(warningsShown[1]['message'], suppression.MODULE_WARNING_MSG)
        self.assertEqual(len(warningsShown), 2)

    def test_suppressModule(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A suppression set on a module prevents warnings emitted by any test\n        mewthods defined in that module which match the suppression from being\n        emitted.\n        '
        self.runTests(self._load(self.TestSuppression2, 'testSuppressModule'))
        warningsShown = self.flushWarnings([self.TestSuppression._emit])
        self.assertEqual(warningsShown[0]['message'], suppression.METHOD_WARNING_MSG)
        self.assertEqual(warningsShown[1]['message'], suppression.CLASS_WARNING_MSG)
        self.assertEqual(len(warningsShown), 2)

    def test_overrideSuppressClass(self):
        if False:
            while True:
                i = 10
        '\n        The suppression set on a test method completely overrides a suppression\n        with wider scope; if it does not match a warning emitted by that test\n        method, the warning is emitted, even if a wider suppression matches.\n        '
        self.runTests(self._load(self.TestSuppression, 'testOverrideSuppressClass'))
        warningsShown = self.flushWarnings([self.TestSuppression._emit])
        self.assertEqual(warningsShown[0]['message'], suppression.METHOD_WARNING_MSG)
        self.assertEqual(warningsShown[1]['message'], suppression.CLASS_WARNING_MSG)
        self.assertEqual(warningsShown[2]['message'], suppression.MODULE_WARNING_MSG)
        self.assertEqual(len(warningsShown), 3)

class SynchronousSuppressionTests(SuppressionMixin, unittest.SynchronousTestCase):
    """
    @see: L{twisted.trial.test.test_tests}
    """
    TestSetUpSuppression = namedAny('twisted.trial.test.suppression.SynchronousTestSetUpSuppression')
    TestTearDownSuppression = namedAny('twisted.trial.test.suppression.SynchronousTestTearDownSuppression')
    TestSuppression = namedAny('twisted.trial.test.suppression.SynchronousTestSuppression')
    TestSuppression2 = namedAny('twisted.trial.test.suppression.SynchronousTestSuppression2')