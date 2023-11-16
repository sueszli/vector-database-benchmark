"""Functional tests for UnicodeScript op."""
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

class UnicodeScriptOpTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testValidScripts(self):
        if False:
            return 10
        inputs = [ord('a'), 1041, 33464, ord(',')]
        with self.cached_session():
            input_vector = constant_op.constant(inputs, dtypes.int32)
            outputs = string_ops.unicode_script(input_vector).eval()
            self.assertAllEqual(outputs, [25, 8, 17, 0])

    @test_util.run_deprecated_v1
    def testInvalidScript(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = [-100, 16777215]
        with self.cached_session():
            input_vector = constant_op.constant(inputs, dtypes.int32)
            outputs = string_ops.unicode_script(input_vector).eval()
            self.assertAllEqual(outputs, [-1, -1])

class UnicodeScriptBenchmarks(test.Benchmark):

    def _generateBenchmarkInput(self, size):
        if False:
            for i in range(10):
                print('nop')
        chars = []
        i = 0
        offset = 0
        continuity_size = 20
        while i < size:
            chars.append(ord('a') + offset)
            i += 1
            offset += 1
            if i % continuity_size == 0:
                offset += 100
                if offset > 129344:
                    offset = 0
        return chars

    def benchmark_unicode_script(self):
        if False:
            for i in range(10):
                print('nop')
        with session.Session(config=benchmark.benchmark_config()) as sess:
            chars = self._generateBenchmarkInput(1000000)
            script = string_ops.unicode_script(chars)
            self.run_op_benchmark(sess, script.op, min_iters=100)
if __name__ == '__main__':
    test.main()