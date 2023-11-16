import unittest
from unittest.test.support import LoggingResult

class Test_FunctionTestCase(unittest.TestCase):

    def test_countTestCases(self):
        if False:
            i = 10
            return i + 15
        test = unittest.FunctionTestCase(lambda : None)
        self.assertEqual(test.countTestCases(), 1)

    def test_run_call_order__error_in_setUp(self):
        if False:
            print('Hello World!')
        events = []
        result = LoggingResult(events)

        def setUp():
            if False:
                while True:
                    i = 10
            events.append('setUp')
            raise RuntimeError('raised by setUp')

        def test():
            if False:
                print('Hello World!')
            events.append('test')

        def tearDown():
            if False:
                return 10
            events.append('tearDown')
        expected = ['startTest', 'setUp', 'addError', 'stopTest']
        unittest.FunctionTestCase(test, setUp, tearDown).run(result)
        self.assertEqual(events, expected)

    def test_run_call_order__error_in_test(self):
        if False:
            while True:
                i = 10
        events = []
        result = LoggingResult(events)

        def setUp():
            if False:
                print('Hello World!')
            events.append('setUp')

        def test():
            if False:
                print('Hello World!')
            events.append('test')
            raise RuntimeError('raised by test')

        def tearDown():
            if False:
                return 10
            events.append('tearDown')
        expected = ['startTest', 'setUp', 'test', 'tearDown', 'addError', 'stopTest']
        unittest.FunctionTestCase(test, setUp, tearDown).run(result)
        self.assertEqual(events, expected)

    def test_run_call_order__failure_in_test(self):
        if False:
            while True:
                i = 10
        events = []
        result = LoggingResult(events)

        def setUp():
            if False:
                for i in range(10):
                    print('nop')
            events.append('setUp')

        def test():
            if False:
                return 10
            events.append('test')
            self.fail('raised by test')

        def tearDown():
            if False:
                print('Hello World!')
            events.append('tearDown')
        expected = ['startTest', 'setUp', 'test', 'tearDown', 'addFailure', 'stopTest']
        unittest.FunctionTestCase(test, setUp, tearDown).run(result)
        self.assertEqual(events, expected)

    def test_run_call_order__error_in_tearDown(self):
        if False:
            while True:
                i = 10
        events = []
        result = LoggingResult(events)

        def setUp():
            if False:
                while True:
                    i = 10
            events.append('setUp')

        def test():
            if False:
                while True:
                    i = 10
            events.append('test')

        def tearDown():
            if False:
                i = 10
                return i + 15
            events.append('tearDown')
            raise RuntimeError('raised by tearDown')
        expected = ['startTest', 'setUp', 'test', 'tearDown', 'addError', 'stopTest']
        unittest.FunctionTestCase(test, setUp, tearDown).run(result)
        self.assertEqual(events, expected)

    def test_id(self):
        if False:
            while True:
                i = 10
        test = unittest.FunctionTestCase(lambda : None)
        self.assertIsInstance(test.id(), str)

    def test_shortDescription__no_docstring(self):
        if False:
            print('Hello World!')
        test = unittest.FunctionTestCase(lambda : None)
        self.assertEqual(test.shortDescription(), None)

    def test_shortDescription__singleline_docstring(self):
        if False:
            return 10
        desc = 'this tests foo'
        test = unittest.FunctionTestCase(lambda : None, description=desc)
        self.assertEqual(test.shortDescription(), 'this tests foo')
if __name__ == '__main__':
    unittest.main()