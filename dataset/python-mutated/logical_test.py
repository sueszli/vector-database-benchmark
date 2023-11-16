"""Tests for logical module."""
from tensorflow.python.autograph.operators import logical
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class LogicalOperatorsTest(test.TestCase):

    def assertNotCalled(self):
        if False:
            print('Hello World!')
        self.fail('this should not be called')

    def _tf_true(self):
        if False:
            return 10
        return constant_op.constant(True)

    def _tf_false(self):
        if False:
            return 10
        return constant_op.constant(False)

    def test_and_python(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(logical.and_(lambda : True, lambda : True))
        self.assertTrue(logical.and_(lambda : [1], lambda : True))
        self.assertListEqual(logical.and_(lambda : True, lambda : [1]), [1])
        self.assertFalse(logical.and_(lambda : False, lambda : True))
        self.assertFalse(logical.and_(lambda : False, self.assertNotCalled))

    @test_util.run_deprecated_v1
    def test_and_tf(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            t = logical.and_(self._tf_true, self._tf_true)
            self.assertEqual(self.evaluate(t), True)
            t = logical.and_(self._tf_true, lambda : True)
            self.assertEqual(self.evaluate(t), True)
            t = logical.and_(self._tf_false, lambda : True)
            self.assertEqual(self.evaluate(t), False)

    def test_or_python(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(logical.or_(lambda : False, lambda : False))
        self.assertFalse(logical.or_(lambda : [], lambda : False))
        self.assertListEqual(logical.or_(lambda : False, lambda : [1]), [1])
        self.assertTrue(logical.or_(lambda : False, lambda : True))
        self.assertTrue(logical.or_(lambda : True, self.assertNotCalled))

    @test_util.run_deprecated_v1
    def test_or_tf(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            t = logical.or_(self._tf_false, self._tf_true)
            self.assertEqual(self.evaluate(t), True)
            t = logical.or_(self._tf_false, lambda : True)
            self.assertEqual(self.evaluate(t), True)
            t = logical.or_(self._tf_true, lambda : True)
            self.assertEqual(self.evaluate(t), True)

    def test_not_python(self):
        if False:
            while True:
                i = 10
        self.assertFalse(logical.not_(True))
        self.assertFalse(logical.not_([1]))
        self.assertTrue(logical.not_([]))

    def test_not_tf(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            t = logical.not_(self._tf_false())
            self.assertEqual(self.evaluate(t), True)
if __name__ == '__main__':
    test.main()