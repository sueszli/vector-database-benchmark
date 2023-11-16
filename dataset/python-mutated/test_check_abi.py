import os
import unittest
import warnings
import paddle.utils.cpp_extension.extension_utils as utils

class TestABIBase(unittest.TestCase):

    def test_environ(self):
        if False:
            return 10
        compiler_list = ['gcc', 'cl']
        for compiler in compiler_list:
            for flag in ['1', 'True', 'true']:
                os.environ['PADDLE_SKIP_CHECK_ABI'] = flag
                self.assertTrue(utils.check_abi_compatibility(compiler))

    def del_environ(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'PADDLE_SKIP_CHECK_ABI'
        if key in os.environ:
            del os.environ[key]

class TestCheckCompiler(TestABIBase):

    def test_expected_compiler(self):
        if False:
            while True:
                i = 10
        if utils.OS_NAME.startswith('linux'):
            gt = ['gcc', 'g++', 'gnu-c++', 'gnu-cc']
        elif utils.IS_WINDOWS:
            gt = ['cl']
        elif utils.OS_NAME.startswith('darwin'):
            gt = ['clang', 'clang++']
        self.assertListEqual(utils._expected_compiler_current_platform(), gt)

    def test_compiler_version(self):
        if False:
            for i in range(10):
                print('nop')
        self.del_environ()
        if utils.OS_NAME.startswith('linux'):
            compiler = 'g++'
        elif utils.IS_WINDOWS:
            compiler = 'cl'
        else:
            compiler = 'clang'
        self.assertTrue(utils.check_abi_compatibility(compiler, verbose=True))

    def test_wrong_compiler_warning(self):
        if False:
            while True:
                i = 10
        self.del_environ()
        compiler = 'python'
        if not utils.IS_WINDOWS:
            with warnings.catch_warnings(record=True) as error:
                flag = utils.check_abi_compatibility(compiler, verbose=True)
                self.assertFalse(flag)
                self.assertTrue(len(error) == 1)
                self.assertTrue('Compiler Compatibility WARNING' in str(error[0].message))

    def test_exception_windows(self):
        if False:
            return 10
        self.del_environ()
        compiler = 'fake compiler'
        if utils.IS_WINDOWS:
            with warnings.catch_warnings(record=True) as error:
                flag = utils.check_abi_compatibility(compiler, verbose=True)
                self.assertFalse(flag)
                self.assertTrue(len(error) == 1)
                self.assertTrue('Failed to check compiler version for' in str(error[0].message))

    def test_exception_linux(self):
        if False:
            print('Hello World!')
        self.del_environ()
        compiler = 'python'
        if utils.OS_NAME.startswith('linux'):

            def fake():
                if False:
                    print('Hello World!')
                return [compiler]
            raw_func = utils._expected_compiler_current_platform
            utils._expected_compiler_current_platform = fake
            with warnings.catch_warnings(record=True) as error:
                flag = utils.check_abi_compatibility(compiler, verbose=True)
                self.assertFalse(flag)
                self.assertTrue(len(error) == 1)
                self.assertTrue('Failed to check compiler version for' in str(error[0].message))
            utils._expected_compiler_current_platform = raw_func

    def test_exception_mac(self):
        if False:
            for i in range(10):
                print('nop')
        self.del_environ()
        compiler = 'python'
        if utils.OS_NAME.startswith('darwin'):

            def fake():
                if False:
                    return 10
                return [compiler]
            raw_func = utils._expected_compiler_current_platform
            utils._expected_compiler_current_platform = fake
            with warnings.catch_warnings(record=True) as error:
                flag = utils.check_abi_compatibility(compiler, verbose=True)
                self.assertTrue(flag)
                self.assertTrue(len(error) == 0)
            utils._expected_compiler_current_platform = raw_func

class TestRunCMDException(unittest.TestCase):

    def test_exception(self):
        if False:
            while True:
                i = 10
        for verbose in [True, False]:
            with self.assertRaisesRegex(RuntimeError, 'Failed to run command'):
                cmd = 'fake cmd'
                utils.run_cmd(cmd, verbose)
if __name__ == '__main__':
    unittest.main()