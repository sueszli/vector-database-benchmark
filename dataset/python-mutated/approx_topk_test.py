"""Tests for approx_max_k and approx_min_k."""
import itertools
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import test
from tensorflow.python.eager.def_function import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables

class ApproxTopkTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            return 10
        test.TestCase.setUp(self)
        self._rng = np.random.default_rng(42)

    def compute_recall(self, result_neighbors, ground_truth_neighbors):
        if False:
            while True:
                i = 10
        'Computes the recall of an approximate nearest neighbor search.\n\n    Args:\n      result_neighbors: int32 numpy array of the shape [num_queries,\n        neighbors_per_query] where the values are the indices of the dataset.\n      ground_truth_neighbors: int32 numpy array of with shape [num_queries,\n        ground_truth_neighbors_per_query] where the values are the indices of\n        the dataset.\n\n    Returns:\n      The recall.\n    '
        self.assertLen(result_neighbors.shape, 2)
        self.assertLen(ground_truth_neighbors.shape, 2)
        self.assertEqual(result_neighbors.shape[0], ground_truth_neighbors.shape[0])
        gt_sets = [set(np.asarray(x)) for x in ground_truth_neighbors]

        def hits_per_q(q, nn_per_q):
            if False:
                return 10
            return len(list((x for x in nn_per_q if x.item() in gt_sets[q])))
        hits = sum((hits_per_q(q, nn_per_q) for (q, nn_per_q) in enumerate(result_neighbors)))
        return hits / ground_truth_neighbors.size

    @parameterized.parameters(itertools.product([1, 10], [100, 500], [1, 10, 128], [True, False]))
    def test_non_fused_max_k(self, k, row_size, num_rows, aggregate_to_topk):
        if False:
            return 10
        row = np.arange(row_size, dtype=np.float32)
        db = np.stack(list((self._rng.permutation(row) for _ in range(num_rows))))

        @function(jit_compile=True)
        def ann(db, k):
            if False:
                for i in range(10):
                    print('nop')
            return nn_ops.approx_max_k(db, k, aggregate_to_topk=aggregate_to_topk)
        with ops.device('/device:TPU:0'):
            db_op = variables.Variable(db)
            result = ann(db_op, k)[1]
        gt = np.argsort(-db)[:, :k]
        ann_recall = self.compute_recall(result.numpy(), gt)
        self.assertGreaterEqual(ann_recall, 0.95)

    @parameterized.parameters(itertools.product([1, 10], [100, 500], [1, 10, 128], [True, False]))
    def test_non_fused_min_k(self, k, row_size, num_rows, aggregate_to_topk):
        if False:
            while True:
                i = 10
        row = np.arange(row_size, dtype=np.float32)
        db = np.stack(list((self._rng.permutation(row) for _ in range(num_rows))))

        @function(jit_compile=True)
        def ann(db, k=10):
            if False:
                i = 10
                return i + 15
            return nn_ops.approx_min_k(db, k, aggregate_to_topk=aggregate_to_topk)
        with ops.device('/device:TPU:0'):
            db_op = variables.Variable(db)
            result = ann(db_op, k)[1]
        gt = np.argsort(db)[:, :k]
        ann_recall = self.compute_recall(result.numpy(), gt)
        self.assertGreaterEqual(ann_recall, 0.95)

    @parameterized.parameters(itertools.product([1, 10], [100, 500], [1, 10, 128], [2, 32]))
    def test_mips(self, k, db_size, qy_size, feature_dim):
        if False:
            i = 10
            return i + 15
        qy = self._rng.random([qy_size, feature_dim], dtype=np.float32)
        db = self._rng.random([db_size, feature_dim], dtype=np.float32)

        @function(jit_compile=True)
        def ann(qy, db, k):
            if False:
                i = 10
                return i + 15
            scores = math_ops.matmul(qy, db, transpose_b=True)
            return nn_ops.approx_max_k(scores, k)
        with ops.device('/device:TPU:0'):
            qy_op = variables.Variable(qy)
            db_op = variables.Variable(db)
            result = ann(qy_op, db_op, k)[1]
            scores = -math_ops.matmul(qy_op, db_op, transpose_b=True)
        gt = np.argsort(scores.numpy())[:, :k]
        ann_recall = self.compute_recall(result.numpy(), gt)
        self.assertGreaterEqual(ann_recall, 0.95)

    @parameterized.parameters(itertools.product([1, 10], [100, 500], [10, 128], [2, 8]))
    def test_l2ann(self, k, db_size, qy_size, feature_dim):
        if False:
            i = 10
            return i + 15
        qy = self._rng.random([qy_size, feature_dim], dtype=np.float32)
        db = self._rng.random([db_size, feature_dim], dtype=np.float32)
        db_half_norm_sq = np.linalg.norm(db, axis=1) ** 2 / 2

        @function(jit_compile=True)
        def ann(qy, db, db_half_norm_sq, k):
            if False:
                print('Hello World!')
            scores = db_half_norm_sq - math_ops.matmul(qy, db, transpose_b=True)
            return nn_ops.approx_min_k(scores, k)
        with ops.device('/device:TPU:0'):
            qy_op = variables.Variable(qy)
            db_op = variables.Variable(db)
            db_half_norm_sq_op = variables.Variable(db_half_norm_sq)
            result = ann(qy_op, db_op, db_half_norm_sq_op, k)[1]
            scores = db_half_norm_sq_op - math_ops.matmul(qy_op, db_op, transpose_b=True)
        gt = np.argsort(scores.numpy())[:, :k]
        ann_recall = self.compute_recall(result.numpy(), gt)
        self.assertGreaterEqual(ann_recall, 0.95)

    def test_highdim(self):
        if False:
            i = 10
            return i + 15
        db = self._rng.random([2, 10, 200, 3], dtype=np.float32)
        k = 5

        @function(jit_compile=True)
        def ann(db, k):
            if False:
                return 10
            return nn_ops.approx_min_k(db, k=k, reduction_dimension=2)
        with ops.device('/device:TPU:0'):
            db_op = variables.Variable(db)
            result = ann(db_op, k)[1]
        gt = np.argsort(db, axis=2)[:, :, :k, :]
        flat_idx = np.reshape(np.transpose(result.numpy(), [0, 1, 3, 2]), [2 * 10 * 3, k])
        flat_gt = np.reshape(np.transpose(gt, [0, 1, 3, 2]), [2 * 10 * 3, k])
        ann_recall = self.compute_recall(flat_idx, flat_gt)
        self.assertGreaterEqual(ann_recall, 0.95)

    @parameterized.parameters(itertools.product([dtypes.bfloat16, dtypes.float16, dtypes.float32], [1, 10], [100, 500], [1, 10, 128]))
    def test_gradients(self, dtype, k, row_size, num_rows):
        if False:
            return 10
        row = np.arange(row_size, dtype=np.float32)
        db = np.stack(list((self._rng.permutation(row) for _ in range(num_rows))))
        out_grads = self._rng.random([num_rows, k])

        @function(jit_compile=True)
        def ann_with_grads(db, out_grads):
            if False:
                return 10
            with backprop.GradientTape() as tape:
                (val, idx) = nn_ops.approx_max_k(db, k)
            result_in_grads = tape.gradient(val, db, out_grads)
            lifted_k_idx = array_ops.reshape(idx, [num_rows, k, 1])
            iota_idx = array_ops.broadcast_to(array_ops.reshape(math_ops.range(num_rows), [num_rows, 1, 1]), [num_rows, k, 1])
            lifted_idx = array_ops.concat([iota_idx, lifted_k_idx], axis=2)
            k_idx_s = array_ops.reshape(lifted_idx, [num_rows * k, 2])
            k_gra_s = array_ops.reshape(out_grads, [num_rows * k])
            expected_in_grads = array_ops.scatter_nd(k_idx_s, k_gra_s, [num_rows, row_size])
            return [expected_in_grads, result_in_grads]
        with ops.device('/device:TPU:0'):
            db_op = variables.Variable(db, dtype=dtype)
            out_grads_op = variables.Variable(out_grads, dtype=dtype)
            (expected_in_grads, result_in_grads) = ann_with_grads(db_op, out_grads_op)
        self.assertAllClose(expected_in_grads, result_in_grads)

    def test_multiple_ops(self):
        if False:
            while True:
                i = 10
        k = 1
        row_size = 100
        num_rows = 10
        row = np.arange(row_size, dtype=np.float32)
        db1 = np.stack(list((self._rng.permutation(row) for _ in range(num_rows))))
        db2 = np.stack(list((self._rng.permutation(row) for _ in range(num_rows))))

        @function(jit_compile=True)
        def ann(db1, db2):
            if False:
                i = 10
                return i + 15
            result1 = nn_ops.approx_max_k(db1, k, aggregate_to_topk=True)
            result2 = nn_ops.approx_max_k(db2, k, aggregate_to_topk=True)
            return (result1, result2)
        with ops.device('/device:TPU:0'):
            db1_op = variables.Variable(db1)
            db2_op = variables.Variable(db2)
            (result1, result2) = ann(db1_op, db2_op)
        gt = np.argsort(-db1)[:, :k]
        ann_recall = self.compute_recall(result1[1].numpy(), gt)
        self.assertGreaterEqual(ann_recall, 0.95)
        gt = np.argsort(-db2)[:, :k]
        ann_recall = self.compute_recall(result2[1].numpy(), gt)
        self.assertGreaterEqual(ann_recall, 0.95)
if __name__ == '__main__':
    test.main()