"""Tests for tensorflow.kernels.logging_ops."""
import sys
from tensorflow.python.framework import test_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

class PrintV2LoggingLevelTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes()
    def testPrintOneTensorLogInfo(self):
        if False:
            return 10
        with self.cached_session():
            tensor = math_ops.range(10)
            with self.captureWritesToStream(sys.stderr) as printed:
                print_op = logging_ops.print_v2(tensor, output_stream=tf_logging.info)
                self.evaluate(print_op)
            self.assertTrue('I' in printed.contents())
            expected = '[0 1 2 ... 7 8 9]'
            self.assertTrue(expected in printed.contents())

    @test_util.run_in_graph_and_eager_modes()
    def testPrintOneTensorLogWarning(self):
        if False:
            return 10
        with self.cached_session():
            tensor = math_ops.range(10)
            with self.captureWritesToStream(sys.stderr) as printed:
                print_op = logging_ops.print_v2(tensor, output_stream=tf_logging.warning)
                self.evaluate(print_op)
            self.assertTrue('W' in printed.contents())
            expected = '[0 1 2 ... 7 8 9]'
            self.assertTrue(expected in printed.contents())

    @test_util.run_in_graph_and_eager_modes()
    def testPrintOneTensorLogError(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            tensor = math_ops.range(10)
            with self.captureWritesToStream(sys.stderr) as printed:
                print_op = logging_ops.print_v2(tensor, output_stream=tf_logging.error)
                self.evaluate(print_op)
            self.assertTrue('E' in printed.contents())
            expected = '[0 1 2 ... 7 8 9]'
            self.assertTrue(expected in printed.contents())
if __name__ == '__main__':
    test.main()