import numpy as np
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class SparseMaskTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testBasic(self):
        if False:
            for i in range(10):
                print('nop')
        values = np.random.rand(4, 4).astype(np.single)
        indices = np.array([0, 2, 3, 4], dtype=np.int32)
        mask_indices = np.array([0], dtype=np.int32)
        out_values = values[1:, :]
        out_indices = np.array([2, 3, 4], dtype=np.int32)
        with self.cached_session() as sess:
            values_tensor = ops.convert_to_tensor(values)
            indices_tensor = ops.convert_to_tensor(indices)
            mask_indices_tensor = ops.convert_to_tensor(mask_indices)
            t = indexed_slices.IndexedSlices(values_tensor, indices_tensor)
            masked_t = array_ops.sparse_mask(t, mask_indices_tensor)
            (tf_out_values, tf_out_indices) = sess.run([masked_t.values, masked_t.indices])
            self.assertAllEqual(tf_out_values, out_values)
            self.assertAllEqual(tf_out_indices, out_indices)
if __name__ == '__main__':
    test.main()