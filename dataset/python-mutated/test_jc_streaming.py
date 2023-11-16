import unittest
import jc.streaming

class MyTests(unittest.TestCase):

    def test_streaming_input_type_check_wrong(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, jc.streaming.streaming_input_type_check, 'abc')

    def test_streaming_input_type_check_correct(self):
        if False:
            while True:
                i = 10
        self.assertEqual(jc.streaming.streaming_input_type_check(['abc']), None)

    def test_streaming_line_input_type_check_wrong(self):
        if False:
            return 10
        self.assertRaises(TypeError, jc.streaming.streaming_line_input_type_check, ['abc'])

    def test_streaming_line_input_type_check_correct(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(jc.streaming.streaming_line_input_type_check('abc'), None)

    def test_stream_success_ignore_exceptions_true(self):
        if False:
            return 10
        self.assertEqual(jc.streaming.stream_success({}, True), {'_jc_meta': {'success': True}})

    def test_stream_success_ignore_exceptions_false(self):
        if False:
            return 10
        self.assertEqual(jc.streaming.stream_success({}, False), {})

    def test_stream_error(self):
        if False:
            return 10
        self.assertEqual(jc.streaming.stream_error(TypeError, 'this is a test'), {'_jc_meta': {'success': False, 'error': "type: <class 'TypeError'>", 'line': 'this is a test'}})

    def test_raise_or_yield_ignore_exceptions(self):
        if False:
            while True:
                i = 10
        self.assertEqual(jc.streaming.raise_or_yield(True, TypeError, 'this is a test'), (TypeError, 'this is a test'))

    def test_raise_or_yield_ignore_exceptions_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, jc.streaming.raise_or_yield, False, TypeError, 'this is a test')
if __name__ == '__main__':
    unittest.main()