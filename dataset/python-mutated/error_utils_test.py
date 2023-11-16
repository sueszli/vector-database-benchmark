"""Tests for error_utils module."""
import re
from tensorflow.python.autograph.pyct import error_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.platform import test

class ErrorMetadataBaseTest(test.TestCase):

    def test_create_exception_default_constructor(self):
        if False:
            while True:
                i = 10

        class CustomError(Exception):
            pass
        em = error_utils.ErrorMetadataBase(callsite_tb=(), cause_metadata=None, cause_message='test message', source_map={}, converter_filename=None)
        exc = em.create_exception(CustomError())
        self.assertIsInstance(exc, CustomError)
        self.assertIn('test message', str(exc))

    def test_create_exception_custom_constructor(self):
        if False:
            print('Hello World!')

        class CustomError(Exception):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(CustomError, self).__init__('test_message')
        em = error_utils.ErrorMetadataBase(callsite_tb=(), cause_metadata=None, cause_message='test message', source_map={}, converter_filename=None)
        exc = em.create_exception(CustomError())
        self.assertIsNone(exc)

    def test_get_message_no_code(self):
        if False:
            i = 10
            return i + 15
        callsite_tb = [('/path/one.py', 11, 'test_fn_1', None), ('/path/two.py', 171, 'test_fn_2', 'test code')]
        cause_message = 'Test message'
        em = error_utils.ErrorMetadataBase(callsite_tb=callsite_tb, cause_metadata=None, cause_message=cause_message, source_map={}, converter_filename=None)
        self.assertRegex(em.get_message(), re.compile('"/path/one.py", line 11, in test_fn_1.*"/path/two.py", line 171, in test_fn_2.*Test message', re.DOTALL))

    def test_get_message_converted_code(self):
        if False:
            while True:
                i = 10
        callsite_tb = [('/path/one.py', 11, 'test_fn_1', 'test code 1'), ('/path/two.py', 171, 'test_fn_2', 'test code 2'), ('/path/three.py', 171, 'test_fn_3', 'test code 3')]
        cause_message = 'Test message'
        em = error_utils.ErrorMetadataBase(callsite_tb=callsite_tb, cause_metadata=None, cause_message=cause_message, source_map={origin_info.LineLocation(filename='/path/two.py', lineno=171): origin_info.OriginInfo(loc=origin_info.LineLocation(filename='/path/other_two.py', lineno=13), function_name='converted_fn', source_code_line='converted test code', comment=None)}, converter_filename=None)
        result = em.get_message()
        self.assertRegex(result, re.compile('converted_fn  \\*.*"/path/three.py", line 171, in test_fn_3.*Test message', re.DOTALL))
        self.assertNotRegex(result, re.compile('test_fn_1'))

    def test_get_message_call_overload(self):
        if False:
            for i in range(10):
                print('nop')
        callsite_tb = [('/path/one.py', 11, 'test_fn_1', 'test code 1'), ('/path/two.py', 0, 'test_fn_2', 'test code 2'), ('/path/three.py', 171, 'test_fn_3', 'test code 3')]
        cause_message = 'Test message'
        em = error_utils.ErrorMetadataBase(callsite_tb=callsite_tb, cause_metadata=None, cause_message=cause_message, source_map={}, converter_filename='/path/two.py')
        self.assertRegex(em.get_message(), re.compile('"/path/one.py", line 11, in test_fn_1.*"/path/three.py", line 171, in test_fn_3  \\*\\*.*Test message', re.DOTALL))
if __name__ == '__main__':
    test.main()