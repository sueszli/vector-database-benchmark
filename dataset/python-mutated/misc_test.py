"""Tests for misc module."""
from tensorflow.python.autograph.utils import misc
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops.variables import Variable
from tensorflow.python.platform import test

class MiscTest(test.TestCase):

    @test_util.run_deprecated_v1
    def test_alias_single_tensor(self):
        if False:
            return 10
        a = constant(1)
        new_a = misc.alias_tensors(a)
        self.assertFalse(new_a is a)
        with self.cached_session() as sess:
            self.assertEqual(1, self.evaluate(new_a))

    @test_util.run_deprecated_v1
    def test_alias_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        a = constant(1)
        v = Variable(2)
        s = 'a'
        l = [1, 2, 3]
        (new_a, new_v, new_s, new_l) = misc.alias_tensors(a, v, s, l)
        self.assertFalse(new_a is a)
        self.assertTrue(new_v is v)
        self.assertTrue(new_s is s)
        self.assertTrue(new_l is l)
        with self.cached_session() as sess:
            self.assertEqual(1, self.evaluate(new_a))

    def test_get_range_len(self):
        if False:
            i = 10
            return i + 15
        get_range_as_graph = def_function.function(misc.get_range_len)
        test_range = [(i, constant_op.constant(i)) for i in range(-3, 3)]
        results = []
        for (i, ti) in test_range:
            for (j, tj) in test_range:
                for (k, tk) in test_range:
                    if k == 0:
                        continue
                    results.append(((i, j, k), get_range_as_graph(ti, tj, tk)))
        for ((i, j, k), result_tensor) in results:
            self.assertEqual(len(list(range(i, j, k))), self.evaluate(result_tensor))
if __name__ == '__main__':
    test.main()