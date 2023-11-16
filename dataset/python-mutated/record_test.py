"""Basic tests for gradients."""
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables

@custom_gradient.custom_gradient
def two_outputs(a, b):
    if False:
        print('Hello World!')
    mm = math_ops.matmul(a, b)
    r = math_ops.reduce_sum(mm)

    def grad(dmm, dr):
        if False:
            i = 10
            return i + 15
        return [math_ops.matmul(dmm, b, transpose_b=True) + math_ops.matmul(array_ops.ones_like(b * dr), b, transpose_b=True), math_ops.matmul(a, dmm, transpose_b=True) + math_ops.matmul(a, array_ops.ones_like(a) * dr, transpose_b=True)]
    return ([mm, r], grad)

@custom_gradient.custom_gradient
def gradient_is_constant(x):
    if False:
        for i in range(10):
            print('nop')
    result = x * x

    def grad(dr):
        if False:
            for i in range(10):
                print('nop')
        return [dr]
    return (result, grad)

class TapeTest(test.TestCase):

    def testMultiOutput(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x, y):
            if False:
                print('Hello World!')
            c = x + y
            (d, f) = array_ops.split(c, 2)
            return d + f
        a = constant_op.constant([[1.0, 0.0], [0.0, 1.0]])
        b = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        (da, db) = backprop.gradients_function(fn, [0, 1])(a, b)
        with context.graph_mode(), self.cached_session():
            tf_a = constant_op.constant([[1, 0], [0, 1]], dtype=dtypes.float32)
            tf_b = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
            tf_c = tf_a + tf_b
            (tf_d, tf_f) = array_ops.split(tf_c, 2, axis=1)
            tf_e = tf_d + tf_f
            (tf_da, tf_db) = gradients_impl.gradients(tf_e, [tf_a, tf_b])
            self.assertAllEqual(da, self.evaluate(tf_da))
            self.assertAllEqual(db, self.evaluate(tf_db))

    def testBasicFunctional(self):
        if False:
            return 10

        def forward(a, b):
            if False:
                i = 10
                return i + 15
            mm = math_ops.matmul(a, b)
            return math_ops.reduce_sum(mm)
        aa = constant_op.constant([[1.0, 0.0], [0.0, 1.0]])
        bb = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        (da,) = backprop.gradients_function(forward, ['a'])(aa, bb)
        self.assertAllEqual(da, math_ops.matmul(array_ops.ones_like(aa), array_ops.transpose(bb)).numpy())

    def testBasicFunctionalPositionalArg(self):
        if False:
            print('Hello World!')

        def forward(a, b):
            if False:
                i = 10
                return i + 15
            mm = math_ops.matmul(a, b)
            return math_ops.reduce_sum(mm)
        aa = constant_op.constant([[1.0, 0.0], [0.0, 1.0]])
        bb = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        (da,) = backprop.gradients_function(forward, [0])(aa, bb)
        self.assertAllEqual(da, math_ops.matmul(array_ops.ones_like(aa), array_ops.transpose(bb)).numpy())

    def testBasicFunctionalWithValue(self):
        if False:
            for i in range(10):
                print('nop')

        def forward(a, b):
            if False:
                for i in range(10):
                    print('nop')
            mm = math_ops.matmul(a, b)
            return math_ops.reduce_sum(mm)
        aa = constant_op.constant([[1.0, 0.0], [0.0, 1.0]])
        bb = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        (val, (da,)) = backprop.val_and_grad_function(forward, ['a'])(aa, bb)
        self.assertAllEqual(da, math_ops.matmul(array_ops.ones_like(aa), array_ops.transpose(bb)))
        self.assertAllEqual(val, forward(aa, bb))

    def testTwoOutputs(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            (mm, r) = two_outputs(x, y)
            return r + math_ops.reduce_sum(mm)
        a = constant_op.constant([[1.0, 0.0], [0.0, 1.0]])
        b = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        (da, db) = backprop.gradients_function(fn, [0, 1])(a, b)
        with context.graph_mode(), self.cached_session():
            tf_a = constant_op.constant([[1, 0], [0, 1]], dtype=dtypes.float32)
            tf_b = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
            tf_mm = math_ops.matmul(tf_a, tf_b)
            tf_rr = 2 * math_ops.reduce_sum(tf_mm)
            (tf_da, tf_db) = gradients_impl.gradients(tf_rr, [tf_a, tf_b])
            self.assertAllEqual(da, self.evaluate(tf_da))
            self.assertAllEqual(db, self.evaluate(tf_db))

    def testGcTwoOutputs(self):
        if False:
            return 10

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y)[0]
        labels = constant_op.constant([0])
        logits = constant_op.constant([[0.0]])
        (grad,) = backprop.gradients_function(fn, [0])(logits, labels)
        self.assertAllEqual(grad, [[0.0]])

    def testTfTensor(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                while True:
                    i = 10
            return x
        t = constant_op.constant(1.0)
        (g,) = backprop.gradients_function(fn, [0])(t)
        self.assertAllEqual(g, 1.0)

class VariableWatcherTest(test.TestCase):

    def testBasic(self):
        if False:
            while True:
                i = 10
        var1 = variables.Variable(0.0)
        var2 = variables.Variable(1.0)
        with record.VariableWatcher() as variable_watcher:
            var1.assign_add(1.0)
            var2.assign_add(2.0)
        self.assertAllEqual(variable_watcher.watched_variables(), (var1, var2))

    def testNonTrainableVariables(self):
        if False:
            for i in range(10):
                print('nop')
        var1 = variables.Variable(0.0)
        var2 = variables.Variable(1.0, trainable=False)
        with record.VariableWatcher() as variable_watcher:
            var1.assign_add(1.0)
            var2.assign_add(2.0)
        self.assertAllEqual(variable_watcher.watched_variables(), (var1,))

    def testMultipleScopes(self):
        if False:
            while True:
                i = 10
        var1 = variables.Variable(0.0)
        var2 = variables.Variable(1.0)
        with record.VariableWatcher() as variable_watcher1:
            var1.assign_add(1.0)
            with record.VariableWatcher() as variable_watcher2:
                var2.assign_add(2.0)
        self.assertAllEqual(variable_watcher1.watched_variables(), (var1, var2))
        self.assertAllEqual(variable_watcher2.watched_variables(), (var2,))

    def testCreateVariables(self):
        if False:
            i = 10
            return i + 15
        with record.VariableWatcher() as variable_watcher:
            var1 = variables.Variable(0.0)
            var2 = variables.Variable(1.0)
            var1.assign_add(1.0)
            var2.assign_add(2.0)
        self.assertAllEqual(variable_watcher.watched_variables(), (var1, var2))
if __name__ == '__main__':
    test.main()