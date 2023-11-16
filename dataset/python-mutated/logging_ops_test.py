"""Tests for tensorflow.kernels.logging_ops."""
import os
import string
import sys
import tempfile
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class LoggingOpsTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testAssertDivideByZero(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            epsilon = ops.convert_to_tensor(1e-20)
            x = ops.convert_to_tensor(0.0)
            y = ops.convert_to_tensor(1.0)
            z = ops.convert_to_tensor(2.0)
            with sess.graph.control_dependencies([control_flow_assert.Assert(math_ops.less(epsilon, y), ['Divide-by-zero'])]):
                out = math_ops.div(z, y)
            self.assertAllEqual(2.0, self.evaluate(out))
            with sess.graph.control_dependencies([control_flow_assert.Assert(math_ops.less(epsilon, x), ['Divide-by-zero', 'less than x'])]):
                out = math_ops.div(z, x)
            with self.assertRaisesOpError('less than x'):
                self.evaluate(out)

@test_util.run_all_in_graph_and_eager_modes
class PrintV2Test(test.TestCase):

    def testPrintOneTensor(self):
        if False:
            for i in range(10):
                print('nop')
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor)
            self.evaluate(print_op)
        expected = '[0 1 2 ... 7 8 9]'
        self.assertIn(expected + '\n', printed.contents())

    def testPrintOneStringTensor(self):
        if False:
            while True:
                i = 10
        tensor = ops.convert_to_tensor([char for char in string.ascii_lowercase])
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor)
            self.evaluate(print_op)
        expected = '["a" "b" "c" ... "x" "y" "z"]'
        self.assertIn(expected + '\n', printed.contents())

    def testPrintOneTensorVarySummarize(self):
        if False:
            print('Hello World!')
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor, summarize=1)
            self.evaluate(print_op)
        expected = '[0 ... 9]'
        self.assertIn(expected + '\n', printed.contents())
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor, summarize=2)
            self.evaluate(print_op)
        expected = '[0 1 ... 8 9]'
        self.assertIn(expected + '\n', printed.contents())
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor, summarize=3)
            self.evaluate(print_op)
        expected = '[0 1 2 ... 7 8 9]'
        self.assertIn(expected + '\n', printed.contents())
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor, summarize=-1)
            self.evaluate(print_op)
        expected = '[0 1 2 3 4 5 6 7 8 9]'
        self.assertIn(expected + '\n', printed.contents())

    def testPrintOneVariable(self):
        if False:
            i = 10
            return i + 15
        var = variables.Variable(math_ops.range(10))
        if not context.executing_eagerly():
            self.evaluate(variables.global_variables_initializer())
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(var)
            self.evaluate(print_op)
        expected = '[0 1 2 ... 7 8 9]'
        self.assertIn(expected + '\n', printed.contents())

    def testPrintTwoVariablesInStructWithAssignAdd(self):
        if False:
            return 10
        var_one = variables.Variable(2.14)
        plus_one = var_one.assign_add(1.0)
        var_two = variables.Variable(math_ops.range(10))
        if not context.executing_eagerly():
            self.evaluate(variables.global_variables_initializer())
        with self.captureWritesToStream(sys.stderr) as printed:
            self.evaluate(plus_one)
            print_op = logging_ops.print_v2(var_one, {'second': var_two})
            self.evaluate(print_op)
        expected = "3.14 {'second': [0 1 2 ... 7 8 9]}"
        self.assertIn(expected + '\n', printed.contents())

    def testPrintTwoTensors(self):
        if False:
            i = 10
            return i + 15
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor, tensor * 10)
            self.evaluate(print_op)
        expected = '[0 1 2 ... 7 8 9] [0 10 20 ... 70 80 90]'
        self.assertIn(expected + '\n', printed.contents())

    def testPrintTwoTensorsDifferentSep(self):
        if False:
            while True:
                i = 10
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor, tensor * 10, sep='<separator>')
            self.evaluate(print_op)
        expected = '[0 1 2 ... 7 8 9]<separator>[0 10 20 ... 70 80 90]'
        self.assertIn(expected + '\n', printed.contents())

    def testPrintPlaceholderGeneration(self):
        if False:
            while True:
                i = 10
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2('{}6', {'{}': tensor * 10})
            self.evaluate(print_op)
        expected = "{}6 {'{}': [0 10 20 ... 70 80 90]}"
        self.assertIn(expected + '\n', printed.contents())

    def testPrintNoTensors(self):
        if False:
            for i in range(10):
                print('nop')
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(23, [23, 5], {'6': 12})
            self.evaluate(print_op)
        expected = "23 [23, 5] {'6': 12}"
        self.assertIn(expected + '\n', printed.contents())

    def testPrintFloatScalar(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.bfloat16, dtypes.half, dtypes.float32, dtypes.float64]:
            tensor = ops.convert_to_tensor(43.5, dtype=dtype)
            with self.captureWritesToStream(sys.stderr) as printed:
                print_op = logging_ops.print_v2(tensor)
                self.evaluate(print_op)
            expected = '43.5'
            self.assertIn(expected + '\n', printed.contents())

    def testPrintStringScalar(self):
        if False:
            while True:
                i = 10
        tensor = ops.convert_to_tensor('scalar')
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor)
            self.evaluate(print_op)
        expected = 'scalar'
        self.assertIn(expected + '\n', printed.contents())

    def testPrintStringScalarDifferentEnd(self):
        if False:
            while True:
                i = 10
        tensor = ops.convert_to_tensor('scalar')
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(tensor, end='<customend>')
            self.evaluate(print_op)
        expected = 'scalar<customend>'
        self.assertIn(expected, printed.contents())

    def testPrintComplexTensorStruct(self):
        if False:
            print('Hello World!')
        tensor = math_ops.range(10)
        small_tensor = constant_op.constant([0.3, 12.4, -16.1])
        big_tensor = math_ops.mul(tensor, 10)
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2('first:', tensor, 'middle:', {'small': small_tensor, 'Big': big_tensor}, 10, [tensor * 2, tensor])
            self.evaluate(print_op)
        expected = "first: [0 1 2 ... 7 8 9] middle: {'Big': [0 10 20 ... 70 80 90], 'small': [0.3 12.4 -16.1]} 10 [[0 2 4 ... 14 16 18], [0 1 2 ... 7 8 9]]"
        self.assertIn(expected + '\n', printed.contents())

    def testPrintSparseTensor(self):
        if False:
            return 10
        ind = [[0, 0], [1, 0], [1, 3], [4, 1], [1, 4], [3, 2], [3, 3]]
        val = [0, 10, 13, 4, 14, 32, 33]
        shape = [5, 6]
        sparse = sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.int64), constant_op.constant(shape, dtypes.int64))
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2(sparse)
            self.evaluate(print_op)
        expected = "'SparseTensor(indices=[[0 0]\n [1 0]\n [1 3]\n ...\n [1 4]\n [3 2]\n [3 3]], values=[0 10 13 ... 14 32 33], shape=[5 6])'"
        self.assertIn(expected + '\n', printed.contents())

    def testPrintSparseTensorInDataStruct(self):
        if False:
            print('Hello World!')
        ind = [[0, 0], [1, 0], [1, 3], [4, 1], [1, 4], [3, 2], [3, 3]]
        val = [0, 10, 13, 4, 14, 32, 33]
        shape = [5, 6]
        sparse = sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.int64), constant_op.constant(shape, dtypes.int64))
        with self.captureWritesToStream(sys.stderr) as printed:
            print_op = logging_ops.print_v2([sparse])
            self.evaluate(print_op)
        expected = "['SparseTensor(indices=[[0 0]\n [1 0]\n [1 3]\n ...\n [1 4]\n [3 2]\n [3 3]], values=[0 10 13 ... 14 32 33], shape=[5 6])']"
        self.assertIn(expected + '\n', printed.contents())

    def testPrintOneTensorStdout(self):
        if False:
            print('Hello World!')
        tensor = math_ops.range(10)
        with self.captureWritesToStream(sys.stdout) as printed:
            print_op = logging_ops.print_v2(tensor, output_stream=sys.stdout)
            self.evaluate(print_op)
        expected = '[0 1 2 ... 7 8 9]'
        self.assertIn(expected + '\n', printed.contents())

    def testPrintTensorsToFile(self):
        if False:
            i = 10
            return i + 15
        (fd, tmpfile_name) = tempfile.mkstemp('.printv2_test')
        tensor_0 = math_ops.range(0, 10)
        print_op_0 = logging_ops.print_v2(tensor_0, output_stream='file://' + tmpfile_name)
        self.evaluate(print_op_0)
        tensor_1 = math_ops.range(11, 20)
        print_op_1 = logging_ops.print_v2(tensor_1, output_stream='file://' + tmpfile_name)
        self.evaluate(print_op_1)
        try:
            f = os.fdopen(fd, 'r')
            line_0 = f.readline()
            expected_0 = '[0 1 2 ... 7 8 9]'
            self.assertTrue(expected_0 in line_0)
            line_1 = f.readline()
            expected_1 = '[11 12 13 ... 17 18 19]'
            self.assertTrue(expected_1 in line_1)
            os.close(fd)
            os.remove(tmpfile_name)
        except IOError as e:
            self.fail(e)

    def testInvalidOutputStreamRaisesError(self):
        if False:
            i = 10
            return i + 15
        tensor = math_ops.range(10)
        with self.assertRaises(ValueError):
            print_op = logging_ops.print_v2(tensor, output_stream='unknown')
            self.evaluate(print_op)

    @test_util.run_deprecated_v1
    def testPrintOpName(self):
        if False:
            for i in range(10):
                print('nop')
        tensor = math_ops.range(10)
        print_op = logging_ops.print_v2(tensor, name='print_name')
        self.assertEqual(print_op.name, 'print_name')

    @test_util.run_deprecated_v1
    def testNoDuplicateFormatOpGraphModeAfterExplicitFormat(self):
        if False:
            return 10
        tensor = math_ops.range(10)
        formatted_string = string_ops.string_format('{}', tensor)
        print_op = logging_ops.print_v2(formatted_string)
        self.evaluate(print_op)
        graph_ops = ops.get_default_graph().get_operations()
        format_ops = [op for op in graph_ops if op.type == 'StringFormat']
        self.assertEqual(len(format_ops), 1)

    def testPrintOneTensorEagerOnOpCreate(self):
        if False:
            for i in range(10):
                print('nop')
        with context.eager_mode():
            tensor = math_ops.range(10)
            expected = '[0 1 2 ... 7 8 9]'
            with self.captureWritesToStream(sys.stderr) as printed:
                logging_ops.print_v2(tensor)
            self.assertIn(expected + '\n', printed.contents())

    def testPrintsOrderedInDefun(self):
        if False:
            while True:
                i = 10
        with context.eager_mode():

            @def_function.function
            def prints():
                if False:
                    for i in range(10):
                        print('nop')
                logging_ops.print_v2('A')
                logging_ops.print_v2('B')
                logging_ops.print_v2('C')
            with self.captureWritesToStream(sys.stderr) as printed:
                prints()
            self.assertTrue('A\nB\nC\n', printed.contents())

    def testPrintInDefunWithoutExplicitEvalOfPrint(self):
        if False:
            return 10
        tensor = math_ops.range(10)

        @def_function.function
        def f(tensor):
            if False:
                i = 10
                return i + 15
            logging_ops.print_v2(tensor)
            return tensor
        expected = '[0 1 2 ... 7 8 9]'
        with self.captureWritesToStream(sys.stderr) as printed_one:
            x = f(tensor)
            self.evaluate(x)
        self.assertIn(expected + '\n', printed_one.contents())
        with self.captureWritesToStream(sys.stderr) as printed_two:
            y = f(tensor)
            self.evaluate(y)
        self.assertIn(expected + '\n', printed_two.contents())

class PrintGradientTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testPrintShape(self):
        if False:
            while True:
                i = 10
        inp = constant_op.constant(2.0, shape=[100, 32])
        inp_printed = logging_ops.Print(inp, [inp])
        self.assertEqual(inp.get_shape(), inp_printed.get_shape())

    def testPrintString(self):
        if False:
            return 10
        inp = constant_op.constant(2.0, shape=[100, 32])
        inp_printed = logging_ops.Print(inp, ['hello'])
        self.assertEqual(inp.get_shape(), inp_printed.get_shape())

    @test_util.run_deprecated_v1
    def testPrintGradient(self):
        if False:
            i = 10
            return i + 15
        inp = constant_op.constant(2.0, shape=[100, 32], name='in')
        w = constant_op.constant(4.0, shape=[10, 100], name='w')
        wx = math_ops.matmul(w, inp, name='wx')
        wx_print = logging_ops.Print(wx, [w, w, w])
        wx_grad = gradients_impl.gradients(wx, w)[0]
        wx_print_grad = gradients_impl.gradients(wx_print, w)[0]
        wxg = self.evaluate(wx_grad)
        wxpg = self.evaluate(wx_print_grad)
        self.assertAllEqual(wxg, wxpg)
if __name__ == '__main__':
    test.main()