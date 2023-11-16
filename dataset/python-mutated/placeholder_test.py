"""Tests for xla handling of placeholder_with_default."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

class PlaceholderTest(xla_test.XLATestCase):

    def test_placeholder_with_default_default(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as sess, self.test_scope():
            v = resource_variable_ops.ResourceVariable(4.0)
            ph = array_ops.placeholder_with_default(v, shape=[])
            out = ph * 2
            sess.run(variables.variables_initializer([v]))
            self.assertEqual(8.0, self.evaluate(out))

    def test_placeholder_with_default_fed(self):
        if False:
            i = 10
            return i + 15
        with self.session() as sess, self.test_scope():
            v = resource_variable_ops.ResourceVariable(4.0)
            ph = array_ops.placeholder_with_default(v, shape=[])
            out = ph * 2
            sess.run(variables.variables_initializer([v]))
            self.assertEqual(2.0, sess.run(out, {ph: 1.0}))
if __name__ == '__main__':
    googletest.main()