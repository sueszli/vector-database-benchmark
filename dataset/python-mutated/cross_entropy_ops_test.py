from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import unittest

def sigmoid(x):
    if False:
        print('Hello World!')
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_cross_entropy_with_logits(x, z):
    if False:
        i = 10
        return i + 15
    return np.maximum(x, 0) - x * z + np.log(1 + np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits_grad(x, z):
    if False:
        i = 10
        return i + 15
    return z - sigmoid(x)

def sigmoid_cross_entropy_with_logits_with_log_D_trick(x, z):
    if False:
        return 10
    return -(2 * z - 1.0) * np.log(sigmoid(x))

def sigmoid_cross_entropy_with_logits_with_log_D_trick_grad(x, z):
    if False:
        return 10
    return (2 * z - 1.0) * (1 - sigmoid(x))

def unjoined_sigmoid_cross_entropy(x, z):
    if False:
        return 10
    return -z * x + (1.0 - z) * np.maximum(x, 0) + (1.0 - z) * np.log(1 + np.exp(-np.abs(x)))

def unjoined_sigmoid_cross_entropy_grad(x, z):
    if False:
        while True:
            i = 10
    return z - (1.0 - z) / (1.0 + np.exp(-x))

class TestCrossEntropyOps(hu.HypothesisTestCase):

    @given(inputs=st.lists(elements=st.integers(min_value=1, max_value=5), min_size=1, max_size=2).flatmap(lambda shape: st.tuples(hu.arrays(dims=shape, elements=st.one_of(hu.floats(min_value=-1.0, max_value=-0.1), hu.floats(min_value=0.1, max_value=1.0))), hu.arrays(dims=shape, elements=st.sampled_from([0.0, 1.0])))), options=st.one_of(st.tuples(st.just(True), st.just(False)), st.tuples(st.just(False), st.just(True)), st.tuples(st.just(False), st.just(False))), **hu.gcs)
    def test_sigmoid_cross_entropy_with_logits(self, inputs, options, gc, dc):
        if False:
            return 10
        (logits, targets) = inputs
        (log_D_trick, unjoined_lr_loss) = options

        def sigmoid_xentr_logit_ref(logits, targets):
            if False:
                i = 10
                return i + 15
            if unjoined_lr_loss:
                s = unjoined_sigmoid_cross_entropy(logits, targets)
            else:
                s = sigmoid_cross_entropy_with_logits(logits, targets) if not log_D_trick else sigmoid_cross_entropy_with_logits_with_log_D_trick(logits, targets)
            m = np.mean(s, axis=len(logits.shape) - 1)
            return (m,)

        def sigmoid_xentr_logit_grad_ref(g_out, outputs, fwd_inputs):
            if False:
                i = 10
                return i + 15
            (fwd_logits, fwd_targets) = fwd_inputs
            inner_size = fwd_logits.shape[-1]
            if unjoined_lr_loss:
                m = unjoined_sigmoid_cross_entropy_grad(logits, targets)
            else:
                m = sigmoid_cross_entropy_with_logits_grad(fwd_logits, fwd_targets) if not log_D_trick else sigmoid_cross_entropy_with_logits_with_log_D_trick_grad(fwd_logits, fwd_targets)
            g_in = -np.expand_dims(g_out, axis=-1) * m / inner_size
            return (g_in, None)
        op = core.CreateOperator('SigmoidCrossEntropyWithLogits', ['logits', 'targets'], ['xentropy'], log_D_trick=log_D_trick, unjoined_lr_loss=unjoined_lr_loss)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[logits, targets], reference=sigmoid_xentr_logit_ref, output_to_grad='xentropy', grad_reference=sigmoid_xentr_logit_grad_ref)

    @given(log_D_trick=st.just(False), **hu.gcs_cpu_only)
    def test_cross_entropy_and_unjoied_cross_entropy_relation(self, log_D_trick, gc, dc):
        if False:
            return 10
        logits = np.array([1.472, 0.35, -0.6529, -1.1908, 0.8357, -1.0774, -0.3395, -0.2469, 0.6708, -1.8332], dtype='f')
        targets = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype='f')
        lr_size = targets.size
        unjoined_lr_loss = False

        def sigmoid_xentr_logit_ref(logits, targets):
            if False:
                for i in range(10):
                    print('nop')
            if unjoined_lr_loss:
                s = unjoined_sigmoid_cross_entropy(logits, targets)
            else:
                s = sigmoid_cross_entropy_with_logits(logits, targets)
            m = np.mean(s, axis=len(logits.shape) - 1)
            return (m,)

        def sigmoid_xentr_logit_grad_ref(g_out, outputs, fwd_inputs):
            if False:
                for i in range(10):
                    print('nop')
            (fwd_logits, fwd_targets) = fwd_inputs
            inner_size = fwd_logits.shape[-1]
            if unjoined_lr_loss:
                m = unjoined_sigmoid_cross_entropy_grad(logits, targets)
            else:
                m = sigmoid_cross_entropy_with_logits_grad(fwd_logits, fwd_targets)
            g_in = -np.expand_dims(g_out, axis=-1) * m / inner_size
            return (g_in, None)
        op = core.CreateOperator('SigmoidCrossEntropyWithLogits', ['logits', 'targets'], ['xentropy'], log_D_trick=log_D_trick, unjoined_lr_loss=unjoined_lr_loss)
        output_lr = self.assertReferenceChecks(device_option=gc, op=op, inputs=[logits, targets], reference=sigmoid_xentr_logit_ref, output_to_grad='xentropy', grad_reference=sigmoid_xentr_logit_grad_ref)
        logits = np.array([1.472, 0.35, -0.6529, -1.1908, 0.8357, -1.0774, -0.3395, -0.2469, 0.6708, -1.8332, 1.472, 0.35, -0.6529, -1.1908, 0.8357, -1.0774], dtype='f')
        targets = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='f')
        unjoined_lr_loss = True
        unjoined_lr_size = targets.size
        op = core.CreateOperator('SigmoidCrossEntropyWithLogits', ['logits', 'targets'], ['xentropy'], log_D_trick=log_D_trick, unjoined_lr_loss=unjoined_lr_loss)
        outputs_unjoined_lr = self.assertReferenceChecks(device_option=gc, op=op, inputs=[logits, targets], reference=sigmoid_xentr_logit_ref, output_to_grad='xentropy', grad_reference=sigmoid_xentr_logit_grad_ref)
        self.assertAlmostEqual(output_lr[0].item(0) * lr_size / unjoined_lr_size, outputs_unjoined_lr[0].item(0), delta=0.0001)

    @given(inputs=st.lists(elements=st.integers(min_value=1, max_value=5), min_size=1, max_size=2).flatmap(lambda shape: st.tuples(hu.arrays(dims=shape, elements=st.one_of(hu.floats(min_value=-1.0, max_value=-0.1), hu.floats(min_value=0.1, max_value=1.0))), hu.arrays(dims=shape, elements=st.sampled_from([0.0, 1.0])), hu.arrays(dims=shape, elements=hu.floats(min_value=0.1, max_value=1.0)))), **hu.gcs)
    def test_weighted_sigmoid_cross_entropy_with_logits(self, inputs, gc, dc):
        if False:
            return 10
        (logits, targets, weights) = inputs

        def weighted_sigmoid_xentr_logit_ref(logits, targets, weights):
            if False:
                while True:
                    i = 10
            s = sigmoid_cross_entropy_with_logits(logits, targets)
            s = np.multiply(s, weights)
            m = np.mean(s, axis=len(logits.shape) - 1)
            return (m,)

        def weighted_sigmoid_xentr_logit_grad_ref(g_out, outputs, fwd_inputs):
            if False:
                i = 10
                return i + 15
            (fwd_logits, fwd_targets, fwd_weights) = fwd_inputs
            inner_size = fwd_logits.shape[-1]
            m = fwd_targets - sigmoid(fwd_logits)
            m = np.multiply(m, weights)
            g_in = -np.expand_dims(g_out, axis=-1) * m / inner_size
            return (g_in, None, None)
        op = core.CreateOperator('WeightedSigmoidCrossEntropyWithLogits', ['logits', 'targets', 'weights'], ['xentropy'])
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[logits, targets, weights], reference=weighted_sigmoid_xentr_logit_ref, output_to_grad='xentropy', grad_reference=weighted_sigmoid_xentr_logit_grad_ref)

    @given(n=st.integers(2, 10), b=st.integers(1, 5), **hu.gcs_cpu_only)
    def test_soft_label_cross_entropy(self, n, b, gc, dc):
        if False:
            print('Hello World!')
        X = np.random.rand(b, n).astype(np.float32)
        X = X + 0.01
        for i in range(b):
            X[i] = X[i] / np.sum(X[i])
        label = np.random.rand(b, n).astype(np.float32)
        for i in range(b):
            label[i] = label[i] / np.sum(label[i])

        def soft_label_xentr_ref(X, label):
            if False:
                print('Hello World!')
            xent = [np.sum((-label[j][i] * np.log(max(X[j][i], 1e-20)) for i in range(len(X[0])))) for j in range(b)]
            return (xent,)
        op = core.CreateOperator('CrossEntropy', ['X', 'label'], ['Y'])
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, label], reference=soft_label_xentr_ref)
        self.assertGradientChecks(gc, op, [X, label], 0, [0], stepsize=0.0001, threshold=0.01)
if __name__ == '__main__':
    import unittest
    unittest.main()