"""
Test of `pika.diagnostic_utils`
"""
import unittest
import logging
import pika.compat
from pika import diagnostic_utils

class DiagnosticUtilsTest(unittest.TestCase):

    def test_args_and_return_value_propagation(self):
        if False:
            while True:
                i = 10
        bucket = []
        log_exception = diagnostic_utils.create_log_exception_decorator(logging.getLogger(__name__))
        return_value = (1, 2, 3)

        @log_exception
        def my_func(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            bucket.append((args, kwargs))
            return return_value
        expected_args = ('a', 2, 'B', Exception('oh-oh'))
        expected_kwargs = dict(hello='world', bye='hello', error=RuntimeError())
        result = my_func(*expected_args, **expected_kwargs)
        self.assertIs(result, return_value)
        self.assertEqual(bucket, [(expected_args, expected_kwargs)])
        for i in pika.compat.xrange(len(expected_args)):
            self.assertIs(bucket[0][0][i], expected_args[i])
        for key in pika.compat.dictkeys(expected_kwargs):
            self.assertIs(bucket[0][1][key], expected_kwargs[key])
        expected_args = tuple()
        expected_kwargs = dict()
        del bucket[:]
        result = my_func()
        self.assertIs(result, return_value)
        self.assertEqual(bucket, [(expected_args, expected_kwargs)])

    def test_exception_propagation(self):
        if False:
            return 10
        logger = logging.getLogger(__name__)
        log_exception = diagnostic_utils.create_log_exception_decorator(logger)
        log_record_bucket = []
        logger.handle = log_record_bucket.append
        exception = Exception('Oops!')

        @log_exception
        def my_func_that_raises():
            if False:
                i = 10
                return i + 15
            raise exception
        with self.assertRaises(Exception) as ctx:
            my_func_that_raises()
        self.assertIs(ctx.exception, exception)
        self.assertEqual(len(log_record_bucket), 1)
        log_record = log_record_bucket[0]
        print(log_record.getMessage())
        expected_ending = 'Exception: Oops!\n'
        self.assertEqual(log_record.getMessage()[-len(expected_ending):], expected_ending)