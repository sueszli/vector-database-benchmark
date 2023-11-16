"""Tests for third_party.tensorflow.python.framework.indexed_slices."""
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import indexed_slices
from tensorflow.python.platform import test

class IndexedSlicesTest(test.TestCase):

    def testCompositeTensorGradient(self):
        if False:
            for i in range(10):
                print('nop')
        i = indexed_slices.IndexedSlices(values=constant_op.constant([[1.0, 2.0]]), indices=constant_op.constant([1]), dense_shape=[3, 2])
        gradient_components = composite_tensor_gradient.get_flat_tensors_for_gradients([i])
        self.assertAllEqual(gradient_components, [i])
        t = [3.0, 4.0]
        result = composite_tensor_gradient.replace_flat_tensors_for_gradients([i], [t])
        self.assertAllEqual(result, [t])
if __name__ == '__main__':
    test.main()