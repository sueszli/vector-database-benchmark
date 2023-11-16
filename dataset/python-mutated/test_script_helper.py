"""Unittests for test.support.script_helper.  Who tests the test helper?"""
import subprocess
import sys
import os
from test.support import script_helper
import unittest
from unittest import mock

class TestScriptHelper(unittest.TestCase):

    def test_assert_python_ok(self):
        if False:
            while True:
                i = 10
        t = script_helper.assert_python_ok('-c', 'import sys; sys.exit(0)')
        self.assertEqual(0, t[0], 'return code was not 0')

    def test_assert_python_failure(self):
        if False:
            return 10
        (rc, out, err) = script_helper.assert_python_failure('-c', 'sys.exit(0)')
        self.assertNotEqual(0, rc, 'return code should not be 0')

    def test_assert_python_ok_raises(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AssertionError) as error_context:
            script_helper.assert_python_ok('-c', 'sys.exit(0)')
        error_msg = str(error_context.exception)
        self.assertIn('command line:', error_msg)
        self.assertIn('sys.exit(0)', error_msg, msg='unexpected command line')

    def test_assert_python_failure_raises(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AssertionError) as error_context:
            script_helper.assert_python_failure('-c', 'import sys; sys.exit(0)')
        error_msg = str(error_context.exception)
        self.assertIn('Process return code is 0\n', error_msg)
        self.assertIn('import sys; sys.exit(0)', error_msg, msg='unexpected command line.')

    @mock.patch('subprocess.Popen')
    def test_assert_python_isolated_when_env_not_required(self, mock_popen):
        if False:
            i = 10
            return i + 15
        with mock.patch.object(script_helper, 'interpreter_requires_environment', return_value=False) as mock_ire_func:
            mock_popen.side_effect = RuntimeError('bail out of unittest')
            try:
                script_helper._assert_python(True, '-c', 'None')
            except RuntimeError as err:
                self.assertEqual('bail out of unittest', err.args[0])
            self.assertEqual(1, mock_popen.call_count)
            self.assertEqual(1, mock_ire_func.call_count)
            popen_command = mock_popen.call_args[0][0]
            self.assertEqual(sys.executable, popen_command[0])
            self.assertIn('None', popen_command)
            self.assertIn('-I', popen_command)
            self.assertNotIn('-E', popen_command)

    @mock.patch('subprocess.Popen')
    def test_assert_python_not_isolated_when_env_is_required(self, mock_popen):
        if False:
            return 10
        'Ensure that -I is not passed when the environment is required.'
        with mock.patch.object(script_helper, 'interpreter_requires_environment', return_value=True) as mock_ire_func:
            mock_popen.side_effect = RuntimeError('bail out of unittest')
            try:
                script_helper._assert_python(True, '-c', 'None')
            except RuntimeError as err:
                self.assertEqual('bail out of unittest', err.args[0])
            popen_command = mock_popen.call_args[0][0]
            self.assertNotIn('-I', popen_command)
            self.assertNotIn('-E', popen_command)

class TestScriptHelperEnvironment(unittest.TestCase):
    """Code coverage for interpreter_requires_environment()."""

    def setUp(self):
        if False:
            return 10
        self.assertTrue(hasattr(script_helper, '__cached_interp_requires_environment'))
        script_helper.__dict__['__cached_interp_requires_environment'] = None

    def tearDown(self):
        if False:
            print('Hello World!')
        script_helper.__dict__['__cached_interp_requires_environment'] = None

    @mock.patch('subprocess.check_call')
    def test_interpreter_requires_environment_true(self, mock_check_call):
        if False:
            print('Hello World!')
        with mock.patch.dict(os.environ):
            os.environ.pop('PYTHONHOME', None)
            mock_check_call.side_effect = subprocess.CalledProcessError('', '')
            self.assertTrue(script_helper.interpreter_requires_environment())
            self.assertTrue(script_helper.interpreter_requires_environment())
            self.assertEqual(1, mock_check_call.call_count)

    @mock.patch('subprocess.check_call')
    def test_interpreter_requires_environment_false(self, mock_check_call):
        if False:
            i = 10
            return i + 15
        with mock.patch.dict(os.environ):
            os.environ.pop('PYTHONHOME', None)
            script_helper.interpreter_requires_environment()
            self.assertFalse(script_helper.interpreter_requires_environment())
            self.assertEqual(1, mock_check_call.call_count)

    @mock.patch('subprocess.check_call')
    def test_interpreter_requires_environment_details(self, mock_check_call):
        if False:
            i = 10
            return i + 15
        with mock.patch.dict(os.environ):
            os.environ.pop('PYTHONHOME', None)
            script_helper.interpreter_requires_environment()
            self.assertFalse(script_helper.interpreter_requires_environment())
            self.assertFalse(script_helper.interpreter_requires_environment())
            self.assertEqual(1, mock_check_call.call_count)
            check_call_command = mock_check_call.call_args[0][0]
            self.assertEqual(sys.executable, check_call_command[0])
            self.assertIn('-E', check_call_command)

    @mock.patch('subprocess.check_call')
    def test_interpreter_requires_environment_with_pythonhome(self, mock_check_call):
        if False:
            i = 10
            return i + 15
        with mock.patch.dict(os.environ):
            os.environ['PYTHONHOME'] = 'MockedHome'
            self.assertTrue(script_helper.interpreter_requires_environment())
            self.assertTrue(script_helper.interpreter_requires_environment())
            self.assertEqual(0, mock_check_call.call_count)
if __name__ == '__main__':
    unittest.main()