import unittest
import warnings
import paddle
import paddle.version as base_version
from paddle import base

class VersionTest(unittest.TestCase):

    def test_check_output(self):
        if False:
            return 10
        warnings.warn('paddle.__version__: {}, base_version.full_version: {}, base_version.major: {}, base_version.minor: {}, base_version.patch: {}, base_version.rc: {}.'.format(paddle.__version__, base_version.full_version, base_version.major, base_version.minor, base_version.patch, base_version.rc))
        ori_full_version = base_version.full_version
        ori_sep_version = [base_version.major, base_version.minor, base_version.patch, base_version.rc]
        [base_version.major, base_version.minor, base_version.patch, base_version.rc] = ['1', '4', '1', '0']
        base.require_version('1')
        base.require_version('1.4')
        base.require_version('1.4.1.0')
        base.require_version('1.4.1')
        base.require_version(min_version='1.4.1', max_version='1.6.0')
        base.require_version(min_version='1.4.1', max_version='1.4.1')
        [base_version.major, base_version.minor, base_version.patch, base_version.rc] = ['0', '0', '0', '0']
        base.require_version('0.0.0')
        base_version.full_version = ori_full_version
        [base_version.major, base_version.minor, base_version.patch, base_version.rc] = ori_sep_version

class TestErrors(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10

        def test_input_type():
            if False:
                print('Hello World!')
            base.require_version(100)
        self.assertRaises(TypeError, test_input_type)

        def test_input_type_1():
            if False:
                i = 10
                return i + 15
            base.require_version('0', 200)
        self.assertRaises(TypeError, test_input_type_1)

        def test_input_value_1():
            if False:
                while True:
                    i = 10
            base.require_version('string')
        self.assertRaises(ValueError, test_input_value_1)

        def test_input_value_1_1():
            if False:
                while True:
                    i = 10
            base.require_version('1.5', 'string')
        self.assertRaises(ValueError, test_input_value_1_1)

        def test_input_value_2():
            if False:
                i = 10
                return i + 15
            base.require_version('1.5.2.0.0')
        self.assertRaises(ValueError, test_input_value_2)

        def test_input_value_2_1():
            if False:
                i = 10
                return i + 15
            base.require_version('1.5', '1.5.2.0.0')
        self.assertRaises(ValueError, test_input_value_2_1)

        def test_input_value_3():
            if False:
                return 10
            base.require_version('1.5.2a.0')
        self.assertRaises(ValueError, test_input_value_3)

        def test_version():
            if False:
                for i in range(10):
                    print('nop')
            base.require_version('100')

        def test_version_1():
            if False:
                i = 10
                return i + 15
            base.require_version('0.0.0', '1.4')

        def test_version_2():
            if False:
                for i in range(10):
                    print('nop')
            base.require_version('1.4.0', '1.2')
        ori_full_version = base_version.full_version
        ori_sep_version = [base_version.major, base_version.minor, base_version.patch, base_version.rc]
        [base_version.major, base_version.minor, base_version.patch, base_version.rc] = ['1', '4', '1', '0']
        self.assertRaises(Exception, test_version)
        self.assertRaises(Exception, test_version_1)
        self.assertRaises(Exception, test_version_2)
        base_version.full_version = ori_full_version
        [base_version.major, base_version.minor, base_version.patch, base_version.rc] = ori_sep_version
if __name__ == '__main__':
    unittest.main()