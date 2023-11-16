"""Unit tests for the processes module."""
import subprocess
import unittest
import mock
from apache_beam.utils import processes

class Exec(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @mock.patch('apache_beam.utils.processes.subprocess')
    def test_method_forwarding_not_windows(self, *unused_mocks):
        if False:
            for i in range(10):
                print('nop')
        processes.force_shell = False
        processes.call(['subprocess', 'call'], shell=False, other_arg=True)
        processes.subprocess.call.assert_called_once_with(['subprocess', 'call'], shell=False, other_arg=True)
        processes.check_call(['subprocess', 'check_call'], shell=False, other_arg=True)
        processes.subprocess.check_call.assert_called_once_with(['subprocess', 'check_call'], shell=False, other_arg=True)
        processes.check_output(['subprocess', 'check_output'], shell=False)
        processes.subprocess.check_output.assert_called_once_with(['subprocess', 'check_output'], shell=False)
        processes.Popen(['subprocess', 'Popen'], shell=False, other_arg=True)
        processes.subprocess.Popen.assert_called_once_with(['subprocess', 'Popen'], shell=False, other_arg=True)

    @mock.patch('apache_beam.utils.processes.subprocess')
    def test_method_forwarding_windows(self, *unused_mocks):
        if False:
            print('Hello World!')
        processes.force_shell = True
        processes.call(['subprocess', 'call'], shell=False, other_arg=True)
        processes.subprocess.call.assert_called_once_with(['subprocess', 'call'], shell=True, other_arg=True)
        processes.check_call(['subprocess', 'check_call'], shell=False, other_arg=True)
        processes.subprocess.check_call.assert_called_once_with(['subprocess', 'check_call'], shell=True, other_arg=True)
        processes.check_output(['subprocess', 'check_output'], shell=False)
        processes.subprocess.check_output.assert_called_once_with(['subprocess', 'check_output'], shell=True)
        processes.Popen(['subprocess', 'Popen'], shell=False, other_arg=True)
        processes.subprocess.Popen.assert_called_once_with(['subprocess', 'Popen'], shell=True, other_arg=True)

class TestErrorHandlingCheckCall(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.mock_get_patcher = mock.patch('apache_beam.utils.processes.subprocess.check_call')
        cls.mock_get = cls.mock_get_patcher.start()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.mock_get_patcher.stop()

    def test_oserror_check_call(self):
        if False:
            while True:
                i = 10
        self.mock_get.side_effect = OSError('Test OSError')
        with self.assertRaises(RuntimeError):
            processes.check_call(['lls'])

    def test_oserror_check_call_message(self):
        if False:
            i = 10
            return i + 15
        self.mock_get.side_effect = OSError()
        cmd = ['lls']
        try:
            processes.check_call(cmd)
        except RuntimeError as error:
            self.assertIn('Executable {} not found'.format(str(cmd)), error.args[0])

    def test_check_call_pip_install_non_existing_package(self):
        if False:
            for i in range(10):
                print('nop')
        returncode = 1
        package = 'non-exsisting-package'
        cmd = ['python', '-m', 'pip', 'download', '--dest', '/var', '{}'.format(package), '--no-deps', '--no-binary', ':all:']
        output = 'Collecting {}'.format(package)
        self.mock_get.side_effect = subprocess.CalledProcessError(returncode, cmd, output=output)
        try:
            output = processes.check_call(cmd)
            self.fail('The test failed due to that        no error was raised when calling process.check_call')
        except RuntimeError as error:
            self.assertIn('Output from execution of subprocess: {}'.format(output), error.args[0])
            self.assertIn('Pip install failed for package: {}'.format(package), error.args[0])

class TestErrorHandlingCheckOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.mock_get_patcher = mock.patch('apache_beam.utils.processes.subprocess.check_output')
        cls.mock_get = cls.mock_get_patcher.start()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.mock_get_patcher.stop()

    def test_oserror_check_output_message(self):
        if False:
            print('Hello World!')
        self.mock_get.side_effect = OSError()
        cmd = ['lls']
        try:
            processes.check_output(cmd)
        except RuntimeError as error:
            self.assertIn('Executable {} not found'.format(str(cmd)), error.args[0])

    def test_check_output_pip_install_non_existing_package(self):
        if False:
            i = 10
            return i + 15
        returncode = 1
        package = 'non-exsisting-package'
        cmd = ['python', '-m', 'pip', 'download', '--dest', '/var', '{}'.format(package), '--no-deps', '--no-binary', ':all:']
        output = 'Collecting {}'.format(package)
        self.mock_get.side_effect = subprocess.CalledProcessError(returncode, cmd, output=output)
        try:
            output = processes.check_output(cmd)
            self.fail('The test failed due to that      no error was raised when calling process.check_call')
        except RuntimeError as error:
            self.assertIn('Output from execution of subprocess: {}'.format(output), error.args[0])
            self.assertIn('Pip install failed for package: {}'.format(package), error.args[0])

class TestErrorHandlingCall(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.mock_get_patcher = mock.patch('apache_beam.utils.processes.subprocess.call')
        cls.mock_get = cls.mock_get_patcher.start()

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        cls.mock_get_patcher.stop()

    def test_oserror_check_output_message(self):
        if False:
            while True:
                i = 10
        self.mock_get.side_effect = OSError()
        cmd = ['lls']
        try:
            processes.call(cmd)
        except RuntimeError as error:
            self.assertIn('Executable {} not found'.format(str(cmd)), error.args[0])

    def test_check_output_pip_install_non_existing_package(self):
        if False:
            print('Hello World!')
        returncode = 1
        package = 'non-exsisting-package'
        cmd = ['python', '-m', 'pip', 'download', '--dest', '/var', '{}'.format(package), '--no-deps', '--no-binary', ':all:']
        output = 'Collecting {}'.format(package)
        self.mock_get.side_effect = subprocess.CalledProcessError(returncode, cmd, output=output)
        try:
            output = processes.call(cmd)
            self.fail('The test failed due to that        no error was raised when calling process.check_call')
        except RuntimeError as error:
            self.assertIn('Output from execution of subprocess: {}'.format(output), error.args[0])
            self.assertIn('Pip install failed for package: {}'.format(package), error.args[0])
if __name__ == '__main__':
    unittest.main()