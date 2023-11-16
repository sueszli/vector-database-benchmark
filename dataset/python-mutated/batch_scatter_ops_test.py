"""Tests for tensorflow.ops.tf.scatter."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

def _AsType(v, vtype):
    if False:
        return 10
    return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)

def _NumpyUpdate(ref, indices, updates):
    if False:
        return 10
    for (i, indx) in np.ndenumerate(indices):
        indx = i[:-1] + (indx,)
        ref[indx] = updates[i]
_TF_OPS_TO_NUMPY = {state_ops.batch_scatter_update: _NumpyUpdate}

class ScatterTest(test.TestCase):

    def _VariableRankTest(self, tf_scatter, vtype, itype, repeat_indices=False, updates_are_scalar=False, method=False):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(8)
        with self.cached_session(use_gpu=False):
            for indices_shape in ((2,), (3, 7), (3, 4, 7)):
                for extra_shape in ((), (5,), (5, 9)):
                    sparse_dim = len(indices_shape) - 1
                    indices = np.random.randint(indices_shape[sparse_dim], size=indices_shape, dtype=itype)
                    updates = _AsType(np.random.randn(*indices_shape + extra_shape), vtype)
                    old = _AsType(np.random.randn(*indices_shape + extra_shape), vtype)
                    new = old.copy()
                    np_scatter = _TF_OPS_TO_NUMPY[tf_scatter]
                    np_scatter(new, indices, updates)
                    ref = variables.Variable(old)
                    self.evaluate(variables.variables_initializer([ref]))
                    if method:
                        ref.batch_scatter_update(indexed_slices.IndexedSlices(indices, updates))
                    else:
                        self.evaluate(tf_scatter(ref, indices, updates))
                    self.assertAllClose(ref, new)

    def testVariableRankUpdate(self):
        if False:
            print('Hello World!')
        vtypes = [np.float32, np.float64]
        for vtype in vtypes:
            for itype in (np.int32, np.int64):
                self._VariableRankTest(state_ops.batch_scatter_update, vtype, itype)

    def testBooleanScatterUpdate(self):
        if False:
            print('Hello World!')
        var = variables.Variable([True, False])
        update0 = state_ops.batch_scatter_update(var, [1], [True])
        update1 = state_ops.batch_scatter_update(var, constant_op.constant([0], dtype=dtypes.int64), [False])
        self.evaluate(variables.variables_initializer([var]))
        self.evaluate([update0, update1])
        self.assertAllEqual([False, True], self.evaluate(var))

    def testScatterOutOfRange(self):
        if False:
            return 10
        params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
        updates = np.array([-3, -4, -5]).astype(np.float32)
        ref = variables.Variable(params)
        self.evaluate(variables.variables_initializer([ref]))
        indices = np.array([2, 0, 5])
        self.evaluate(state_ops.batch_scatter_update(ref, indices, updates))
        indices = np.array([-1, 0, 5])
        with self.assertRaisesOpError('indices\\[0\\] = \\[-1\\] does not index into shape \\[6\\]'):
            self.evaluate(state_ops.batch_scatter_update(ref, indices, updates))
        indices = np.array([2, 0, 6])
        with self.assertRaisesOpError('indices\\[2\\] = \\[6\\] does not index into shape \\[6\\]'):
            self.evaluate(state_ops.batch_scatter_update(ref, indices, updates))
if __name__ == '__main__':
    test.main()