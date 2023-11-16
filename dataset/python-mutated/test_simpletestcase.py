import unittest
from io import StringIO
from unittest import mock
from unittest.suite import _DebugResult
from django.test import SimpleTestCase

class ErrorTestCase(SimpleTestCase):

    def raising_test(self):
        if False:
            while True:
                i = 10
        self._pre_setup.assert_called_once_with()
        raise Exception('debug() bubbles up exceptions before cleanup.')

    def simple_test(self):
        if False:
            return 10
        self._pre_setup.assert_called_once_with()

    @unittest.skip('Skip condition.')
    def skipped_test(self):
        if False:
            i = 10
            return i + 15
        pass

@mock.patch.object(ErrorTestCase, '_post_teardown')
@mock.patch.object(ErrorTestCase, '_pre_setup')
class DebugInvocationTests(SimpleTestCase):

    def get_runner(self):
        if False:
            i = 10
            return i + 15
        return unittest.TextTestRunner(stream=StringIO())

    def isolate_debug_test(self, test_suite, result):
        if False:
            for i in range(10):
                print('nop')
        test_suite._tearDownPreviousClass(None, result)
        test_suite._handleModuleTearDown(result)

    def test_run_cleanup(self, _pre_setup, _post_teardown):
        if False:
            for i in range(10):
                print('nop')
        'Simple test run: catches errors and runs cleanup.'
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('raising_test'))
        result = self.get_runner()._makeResult()
        self.assertEqual(result.errors, [])
        test_suite.run(result)
        self.assertEqual(len(result.errors), 1)
        (_, traceback) = result.errors[0]
        self.assertIn('Exception: debug() bubbles up exceptions before cleanup.', traceback)
        _pre_setup.assert_called_once_with()
        _post_teardown.assert_called_once_with()

    def test_run_pre_setup_error(self, _pre_setup, _post_teardown):
        if False:
            return 10
        _pre_setup.side_effect = Exception('Exception in _pre_setup.')
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('simple_test'))
        result = self.get_runner()._makeResult()
        self.assertEqual(result.errors, [])
        test_suite.run(result)
        self.assertEqual(len(result.errors), 1)
        (_, traceback) = result.errors[0]
        self.assertIn('Exception: Exception in _pre_setup.', traceback)
        _pre_setup.assert_called_once_with()
        self.assertFalse(_post_teardown.called)

    def test_run_post_teardown_error(self, _pre_setup, _post_teardown):
        if False:
            print('Hello World!')
        _post_teardown.side_effect = Exception('Exception in _post_teardown.')
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('simple_test'))
        result = self.get_runner()._makeResult()
        self.assertEqual(result.errors, [])
        test_suite.run(result)
        self.assertEqual(len(result.errors), 1)
        (_, traceback) = result.errors[0]
        self.assertIn('Exception: Exception in _post_teardown.', traceback)
        _pre_setup.assert_called_once_with()
        _post_teardown.assert_called_once_with()

    def test_run_skipped_test_no_cleanup(self, _pre_setup, _post_teardown):
        if False:
            return 10
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('skipped_test'))
        try:
            test_suite.run(self.get_runner()._makeResult())
        except unittest.SkipTest:
            self.fail('SkipTest should not be raised at this stage.')
        self.assertFalse(_post_teardown.called)
        self.assertFalse(_pre_setup.called)

    def test_debug_cleanup(self, _pre_setup, _post_teardown):
        if False:
            return 10
        'Simple debug run without errors.'
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('simple_test'))
        test_suite.debug()
        _pre_setup.assert_called_once_with()
        _post_teardown.assert_called_once_with()

    def test_debug_bubbles_error(self, _pre_setup, _post_teardown):
        if False:
            return 10
        'debug() bubbles up exceptions before cleanup.'
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('raising_test'))
        msg = 'debug() bubbles up exceptions before cleanup.'
        with self.assertRaisesMessage(Exception, msg):
            result = _DebugResult()
            test_suite.run(result, debug=True)
        _pre_setup.assert_called_once_with()
        self.assertFalse(_post_teardown.called)
        self.isolate_debug_test(test_suite, result)

    def test_debug_bubbles_pre_setup_error(self, _pre_setup, _post_teardown):
        if False:
            i = 10
            return i + 15
        'debug() bubbles up exceptions during _pre_setup.'
        msg = 'Exception in _pre_setup.'
        _pre_setup.side_effect = Exception(msg)
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('simple_test'))
        with self.assertRaisesMessage(Exception, msg):
            result = _DebugResult()
            test_suite.run(result, debug=True)
        _pre_setup.assert_called_once_with()
        self.assertFalse(_post_teardown.called)
        self.isolate_debug_test(test_suite, result)

    def test_debug_bubbles_post_teardown_error(self, _pre_setup, _post_teardown):
        if False:
            return 10
        'debug() bubbles up exceptions during _post_teardown.'
        msg = 'Exception in _post_teardown.'
        _post_teardown.side_effect = Exception(msg)
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('simple_test'))
        with self.assertRaisesMessage(Exception, msg):
            result = _DebugResult()
            test_suite.run(result, debug=True)
        _pre_setup.assert_called_once_with()
        _post_teardown.assert_called_once_with()
        self.isolate_debug_test(test_suite, result)

    def test_debug_skipped_test_no_cleanup(self, _pre_setup, _post_teardown):
        if False:
            i = 10
            return i + 15
        test_suite = unittest.TestSuite()
        test_suite.addTest(ErrorTestCase('skipped_test'))
        with self.assertRaisesMessage(unittest.SkipTest, 'Skip condition.'):
            result = _DebugResult()
            test_suite.run(result, debug=True)
        self.assertFalse(_post_teardown.called)
        self.assertFalse(_pre_setup.called)
        self.isolate_debug_test(test_suite, result)