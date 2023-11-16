"""Reference tests check that a function is compiled correctly."""
import io
import numbers
import os
import sys
import traceback
import numpy as np
import tensorflow as tf

class TestCase(tf.test.TestCase):
    """Base class for the reference tests."""

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestCase, self).setUp()
        os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'
        self.autograph_opts = None
        self.all_inputs_tensors = False
        self.allow_exceptions = False

    def _run_with_output_capture(self, func):
        if False:
            return 10
        'Executes `func`, capturing stdout.'
        out_capturer = io.StringIO()
        results = None
        captured_out = None
        captured_err = None
        try:
            sys.stdout = out_capturer
            results = func()
            captured_out = out_capturer.getvalue()
        except Exception as e:
            sys.stdout = sys.__stdout__
            captured_err = e
            print('*** Capturing exception:\n{}\n'.format(traceback.format_exc()))
        finally:
            sys.stdout = sys.__stdout__
            out_capturer.close()
        return (results, captured_out, captured_err)

    def _as_tensors(self, args):
        if False:
            return 10
        'Converts args to tensors.'
        tensor_args = []
        for a in args:
            if isinstance(a, (numbers.Number, list, np.ndarray)):
                tensor_arg = tf.constant(a)
            elif isinstance(a, dict):
                keys = tuple(a.keys())
                tensor_arg = dict(zip(keys, self._as_tensors([a[k] for k in keys])))
            else:
                tensor_arg = a
            tensor_args.append(tensor_arg)
        return tensor_args

    def run_native(self, f, *args):
        if False:
            while True:
                i = 10
        return self._run_with_output_capture(lambda : f(*args))

    def _deep_equal(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        'Compares two possibly-nested structures.'
        if isinstance(left, tf.Tensor):
            return self._deep_equal(left.numpy(), right)
        if isinstance(right, tf.Tensor):
            return self._deep_equal(left, right.numpy())
        if isinstance(left, tf.SparseTensor) and isinstance(right, tf.SparseTensor):
            return self._deep_equal(left.indices, right.indices) and self._deep_equal(left.values, right.values) and self._deep_equal(left.shape, right.shape)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return np.array_equal(left, right)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return all((self._deep_equal(l, r) for (l, r) in zip(left, right)))
        return left == right

    def assertResultsMatch(self, f, args, native_data, compiled_data):
        if False:
            return 10
        'Asserts that native_data matches compiled_data.'
        (native_results, native_out, native_err) = native_data
        (compiled_results, compiled_out, compiled_err) = compiled_data
        str_args = '(%s)' % ', '.join((str(a) for a in args))
        outputs_equal = self._deep_equal(native_results, compiled_results) and native_out == compiled_out
        errors_equivalent = type(native_err) == type(compiled_err)
        if not outputs_equal or not errors_equivalent:
            self.fail('Native and compiled functions are not equivalent.\n\nNative results: %s\nCompiled results: %s\nNative out: %s\nCompiled out: %s\nNative error: %s: %s\nCompiled error: %s: %s\nNative call: %s%s\nCheck the logs for the generated code.' % (native_results, compiled_results, native_out, compiled_out, type(native_err).__name__, native_err, type(compiled_err).__name__, compiled_err, f.__name__, str_args))

    def function(self, f, xla=False):
        if False:
            print('Hello World!')
        return tf.function(f, experimental_autograph_options=self.autograph_opts, experimental_compile=xla)

    def convert(self, f):
        if False:
            i = 10
            return i + 15
        return tf.autograph.to_graph(f, experimental_optional_features=self.autograph_opts)

    def assertFunctionMatchesEagerStatefulInput(self, f, args):
        if False:
            for i in range(10):
                print('nop')
        'Like assertFunctionMatchesEager but creates new inputs each time.'
        compiled_data = self.run_native(self.function(f), *args())
        native_data = self.run_native(f, *args())
        self.assertResultsMatch(f, args(), native_data, compiled_data)

    def assertFunctionMatchesEager(self, f, *args, xla=False):
        if False:
            while True:
                i = 10
        if self.all_inputs_tensors:
            args = self._as_tensors(args)
        compiled_data = self.run_native(self.function(f, xla=xla), *args)
        if not self.allow_exceptions:
            (_, _, compiled_err) = compiled_data
            if compiled_err is not None:
                self.fail(str(compiled_err))
        native_data = self.run_native(f, *args)
        self.assertResultsMatch(f, args, native_data, compiled_data)

    def assertConvertedMatchesNative(self, f, *args):
        if False:
            i = 10
            return i + 15
        compiled_data = self.run_native(self.convert(f), *args)
        native_data = self.run_native(f, *args)
        self.assertResultsMatch(f, args, native_data, compiled_data)
if __name__ == '__main__':
    tf.test.main()