"""Tests for string_join_op."""
from tensorflow.python.framework import errors
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class StringJoinOpTest(test.TestCase):

    def testStringJoin(self):
        if False:
            return 10
        input0 = ['a', 'b']
        input1 = 'a'
        input2 = [['b'], ['c']]
        output = string_ops.string_join([input0, input1])
        self.assertAllEqual(output, [b'aa', b'ba'])
        output = string_ops.string_join([input0, input1], separator='--')
        self.assertAllEqual(output, [b'a--a', b'b--a'])
        output = string_ops.string_join([input0, input1, input0], separator='--')
        self.assertAllEqual(output, [b'a--a--a', b'b--a--b'])
        output = string_ops.string_join([input1] * 4, separator='!')
        self.assertEqual(self.evaluate(output), b'a!a!a!a')
        output = string_ops.string_join([input2] * 2, separator='')
        self.assertAllEqual(output, [[b'bb'], [b'cc']])
        output = string_ops.string_join([])
        self.assertAllEqual(output, b'')
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'shapes do not match|must be equal rank'):
            self.evaluate(string_ops.string_join([input0, input2]))
if __name__ == '__main__':
    test.main()