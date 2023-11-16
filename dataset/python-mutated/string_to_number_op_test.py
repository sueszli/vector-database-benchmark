"""Tests for StringToNumber op from parsing_ops."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
_ERROR_MESSAGE = 'StringToNumberOp could not correctly convert string: '

class StringToNumberOpTest(test.TestCase):

    def _test(self, tf_type, good_pairs, bad_pairs):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            input_string = array_ops.placeholder(dtypes.string)
            output = parsing_ops.string_to_number(input_string, out_type=tf_type)
            for (instr, outnum) in good_pairs:
                (result,) = output.eval(feed_dict={input_string: [instr]})
                self.assertAllClose([outnum], [result])
            for (instr, outstr) in bad_pairs:
                with self.assertRaisesOpError(outstr):
                    output.eval(feed_dict={input_string: [instr]})

    @test_util.run_deprecated_v1
    def testToFloat(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(dtypes.float32, [('0', 0), ('3', 3), ('-1', -1), ('1.12', 1.12), ('0xF', 15), ('   -10.5', -10.5), ('3.40282e+38', 3.40282e+38), ('3.40283e+38', float('INF')), ('-3.40283e+38', float('-INF')), ('NAN', float('NAN')), ('INF', float('INF'))], [('10foobar', _ERROR_MESSAGE + '10foobar')])

    @test_util.run_deprecated_v1
    def testToDouble(self):
        if False:
            print('Hello World!')
        self._test(dtypes.float64, [('0', 0), ('3', 3), ('-1', -1), ('1.12', 1.12), ('0xF', 15), ('   -10.5', -10.5), ('3.40282e+38', 3.40282e+38), ('3.40283e+38', 3.40283e+38), ('-3.40283e+38', -3.40283e+38), ('NAN', float('NAN')), ('INF', float('INF'))], [('10foobar', _ERROR_MESSAGE + '10foobar')])

    @test_util.run_deprecated_v1
    def testToInt32(self):
        if False:
            while True:
                i = 10
        self._test(dtypes.int32, [('0', 0), ('3', 3), ('-1', -1), ('    -10', -10), ('-2147483648', -2147483648), ('2147483647', 2147483647)], [('-2147483649', _ERROR_MESSAGE + '-2147483649'), ('2147483648', _ERROR_MESSAGE + '2147483648'), ('2.9', _ERROR_MESSAGE + '2.9'), ('10foobar', _ERROR_MESSAGE + '10foobar')])

    @test_util.run_deprecated_v1
    def testToInt64(self):
        if False:
            i = 10
            return i + 15
        self._test(dtypes.int64, [('0', 0), ('3', 3), ('-1', -1), ('    -10', -10), ('-2147483648', -2147483648), ('2147483647', 2147483647), ('-2147483649', -2147483649), ('2147483648', 2147483648)], [('2.9', _ERROR_MESSAGE + '2.9'), ('10foobar', _ERROR_MESSAGE + '10foobar')])
if __name__ == '__main__':
    test.main()