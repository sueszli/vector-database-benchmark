import io
import logging
import os
import sys
import unittest
from unittest import mock
import paddle
from paddle.jit.dy2static import logging_utils
from paddle.utils import gast

class TestLoggingUtils(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.verbosity_level = 1
        self.code_level = 3
        self.translator_logger = logging_utils._TRANSLATOR_LOGGER

    def test_verbosity(self):
        if False:
            i = 10
            return i + 15
        paddle.jit.set_verbosity(None)
        os.environ[logging_utils.VERBOSITY_ENV_NAME] = '3'
        self.assertEqual(logging_utils.get_verbosity(), 3)
        paddle.jit.set_verbosity(self.verbosity_level)
        self.assertEqual(self.verbosity_level, logging_utils.get_verbosity())
        with self.assertRaises(TypeError):
            paddle.jit.set_verbosity('3')
        with self.assertRaises(TypeError):
            paddle.jit.set_verbosity(3.3)

    def test_also_to_stdout(self):
        if False:
            i = 10
            return i + 15
        logging_utils._TRANSLATOR_LOGGER.need_to_echo_log_to_stdout = None
        self.assertEqual(logging_utils._TRANSLATOR_LOGGER.need_to_echo_log_to_stdout, False)
        paddle.jit.set_verbosity(also_to_stdout=False)
        self.assertEqual(logging_utils._TRANSLATOR_LOGGER.need_to_echo_log_to_stdout, False)
        logging_utils._TRANSLATOR_LOGGER.need_to_echo_node_to_stdout = None
        self.assertEqual(logging_utils._TRANSLATOR_LOGGER.need_to_echo_code_to_stdout, False)
        paddle.jit.set_code_level(also_to_stdout=True)
        self.assertEqual(logging_utils._TRANSLATOR_LOGGER.need_to_echo_code_to_stdout, True)
        with self.assertRaises(AssertionError):
            paddle.jit.set_verbosity(also_to_stdout=1)
        with self.assertRaises(AssertionError):
            paddle.jit.set_code_level(also_to_stdout=1)

    def test_set_code_level(self):
        if False:
            i = 10
            return i + 15
        paddle.jit.set_code_level(None)
        os.environ[logging_utils.CODE_LEVEL_ENV_NAME] = '2'
        self.assertEqual(logging_utils.get_code_level(), 2)
        paddle.jit.set_code_level(self.code_level)
        self.assertEqual(logging_utils.get_code_level(), self.code_level)
        paddle.jit.set_code_level(9)
        self.assertEqual(logging_utils.get_code_level(), 9)
        with self.assertRaises(TypeError):
            paddle.jit.set_code_level(3.3)

    def test_log_api(self):
        if False:
            for i in range(10):
                print('nop')
        logging_utils.set_verbosity(1, True)
        logging_utils.warn('warn')
        logging_utils.error('error')
        logging_utils.log(1, 'log level 1')
        logging_utils.log(2, 'log level 2')
        source_code = 'x = 3'
        ast_code = gast.parse(source_code)
        logging_utils.set_code_level(1, True)
        logging_utils.log_transformed_code(1, ast_code, 'TestTransformer')
        logging_utils.set_code_level(logging_utils.LOG_AllTransformer, True)
        logging_utils.log_transformed_code(logging_utils.LOG_AllTransformer, ast_code, 'TestTransformer')

    def test_log_message(self):
        if False:
            i = 10
            return i + 15
        stream = io.StringIO()
        log = self.translator_logger.logger
        stdout_handler = logging.StreamHandler(stream)
        log.addHandler(stdout_handler)
        warn_msg = 'test_warn'
        error_msg = 'test_error'
        log_msg_1 = 'test_log_1'
        log_msg_2 = 'test_log_2'
        with mock.patch.object(sys, 'stdout', stream):
            logging_utils.set_verbosity(1, False)
            logging_utils.warn(warn_msg)
            logging_utils.error(error_msg)
            logging_utils.log(1, log_msg_1)
            logging_utils.log(2, log_msg_2)
        result_msg = '\n'.join([warn_msg, error_msg, '(Level 1) ' + log_msg_1, ''])
        self.assertEqual(result_msg, stream.getvalue())

    def test_log_transformed_code(self):
        if False:
            for i in range(10):
                print('nop')
        source_code = 'x = 3'
        ast_code = gast.parse(source_code)
        stream = io.StringIO()
        log = self.translator_logger.logger
        stdout_handler = logging.StreamHandler(stream)
        log.addHandler(stdout_handler)
        with mock.patch.object(sys, 'stdout', stream):
            paddle.jit.set_code_level(1)
            logging_utils.log_transformed_code(1, ast_code, 'BasicApiTransformer')
            paddle.jit.set_code_level()
            logging_utils.log_transformed_code(logging_utils.LOG_AllTransformer, ast_code, 'All Transformers')
        self.assertIn(source_code, stream.getvalue())
if __name__ == '__main__':
    unittest.main()