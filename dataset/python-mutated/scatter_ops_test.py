"""Tests for tensorflow.ops.tf.scatter."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

def _AsType(v, vtype):
    if False:
        for i in range(10):
            print('nop')
    return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)

def _NumpyAdd(ref, indices, updates):
    if False:
        for i in range(10):
            print('nop')
    for (i, indx) in np.ndenumerate(indices):
        ref[indx] += updates[i]

def _NumpyAddScalar(ref, indices, update):
    if False:
        print('Hello World!')
    for (_, indx) in np.ndenumerate(indices):
        ref[indx] += update

def _NumpySub(ref, indices, updates):
    if False:
        print('Hello World!')
    for (i, indx) in np.ndenumerate(indices):
        ref[indx] -= updates[i]

def _NumpySubScalar(ref, indices, update):
    if False:
        print('Hello World!')
    for (_, indx) in np.ndenumerate(indices):
        ref[indx] -= update

def _NumpyMul(ref, indices, updates):
    if False:
        print('Hello World!')
    for (i, indx) in np.ndenumerate(indices):
        ref[indx] *= updates[i]

def _NumpyMulScalar(ref, indices, update):
    if False:
        i = 10
        return i + 15
    for (_, indx) in np.ndenumerate(indices):
        ref[indx] *= update

def _NumpyDiv(ref, indices, updates):
    if False:
        print('Hello World!')
    for (i, indx) in np.ndenumerate(indices):
        ref[indx] /= updates[i]

def _NumpyDivScalar(ref, indices, update):
    if False:
        for i in range(10):
            print('nop')
    for (_, indx) in np.ndenumerate(indices):
        ref[indx] /= update

def _NumpyMin(ref, indices, updates):
    if False:
        for i in range(10):
            print('nop')
    for (i, indx) in np.ndenumerate(indices):
        ref[indx] = np.minimum(ref[indx], updates[i])

def _NumpyMinScalar(ref, indices, update):
    if False:
        for i in range(10):
            print('nop')
    for (_, indx) in np.ndenumerate(indices):
        ref[indx] = np.minimum(ref[indx], update)

def _NumpyMax(ref, indices, updates):
    if False:
        print('Hello World!')
    for (i, indx) in np.ndenumerate(indices):
        ref[indx] = np.maximum(ref[indx], updates[i])

def _NumpyMaxScalar(ref, indices, update):
    if False:
        i = 10
        return i + 15
    for (_, indx) in np.ndenumerate(indices):
        ref[indx] = np.maximum(ref[indx], update)

def _NumpyUpdate(ref, indices, updates):
    if False:
        return 10
    for (i, indx) in np.ndenumerate(indices):
        ref[indx] = updates[i]

def _NumpyUpdateScalar(ref, indices, update):
    if False:
        return 10
    for (_, indx) in np.ndenumerate(indices):
        ref[indx] = update
_TF_OPS_TO_NUMPY = {state_ops.scatter_update: _NumpyUpdate, state_ops.scatter_add: _NumpyAdd, state_ops.scatter_sub: _NumpySub, state_ops.scatter_mul: _NumpyMul, state_ops.scatter_div: _NumpyDiv, state_ops.scatter_min: _NumpyMin, state_ops.scatter_max: _NumpyMax}
_TF_OPS_TO_NUMPY_SCALAR = {state_ops.scatter_update: _NumpyUpdateScalar, state_ops.scatter_add: _NumpyAddScalar, state_ops.scatter_sub: _NumpySubScalar, state_ops.scatter_mul: _NumpyMulScalar, state_ops.scatter_div: _NumpyDivScalar, state_ops.scatter_min: _NumpyMinScalar, state_ops.scatter_max: _NumpyMaxScalar}

class ScatterTest(test.TestCase):

    def _VariableRankTest(self, tf_scatter, vtype, itype, repeat_indices=False, updates_are_scalar=False):
        if False:
            return 10
        np.random.seed(8)
        with self.cached_session():
            for indices_shape in ((), (2,), (3, 7), (3, 4, 7)):
                for extra_shape in ((), (5,), (5, 9)):
                    size = np.prod(indices_shape, dtype=itype)
                    first_dim = 3 * size
                    indices = np.arange(first_dim)
                    np.random.shuffle(indices)
                    indices = indices[:size]
                    if size > 1 and repeat_indices:
                        indices = indices[:size // 2]
                        for _ in range(size - size // 2):
                            indices = np.append(indices, indices[np.random.randint(size // 2)])
                        np.random.shuffle(indices)
                    indices = indices.reshape(indices_shape)
                    if updates_are_scalar:
                        updates = _AsType(np.random.randn(), vtype)
                    else:
                        updates = _AsType(np.random.randn(*indices_shape + extra_shape), vtype)
                    threshold = np.array(0.0001, dtype=vtype)
                    sign = np.sign(updates)
                    if vtype == np.int32:
                        threshold = 1
                        sign = np.random.choice([-1, 1], updates.shape)
                    updates = np.where(np.abs(updates) < threshold, threshold * sign, updates)
                    old = _AsType(np.random.randn(*(first_dim,) + extra_shape), vtype)
                    new = old.copy()
                    if updates_are_scalar:
                        np_scatter = _TF_OPS_TO_NUMPY_SCALAR[tf_scatter]
                    else:
                        np_scatter = _TF_OPS_TO_NUMPY[tf_scatter]
                    np_scatter(new, indices, updates)
                    ref = variables.Variable(old)
                    self.evaluate(ref.initializer)
                    self.evaluate(tf_scatter(ref, indices, updates))
                    self.assertAllCloseAccordingToType(self.evaluate(ref), new, half_rtol=0.005, half_atol=0.005, bfloat16_rtol=0.05, bfloat16_atol=0.05)

    def _VariableRankTests(self, tf_scatter, repeat_indices=False, updates_are_scalar=False):
        if False:
            while True:
                i = 10
        vtypes = [np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]
        if tf_scatter != state_ops.scatter_div:
            vtypes.append(np.int32)
            vtypes.append(np.float16)
        for vtype in vtypes:
            for itype in (np.int32, np.int64):
                self._VariableRankTest(tf_scatter, vtype, itype, repeat_indices, updates_are_scalar)

    def testVariableRankUpdate(self):
        if False:
            return 10
        self._VariableRankTests(state_ops.scatter_update, False)

    def testVariableRankAdd(self):
        if False:
            return 10
        self._VariableRankTests(state_ops.scatter_add, False)

    def testVariableRankSub(self):
        if False:
            while True:
                i = 10
        self._VariableRankTests(state_ops.scatter_sub, False)

    def testVariableRankMul(self):
        if False:
            print('Hello World!')
        self._VariableRankTests(state_ops.scatter_mul, False)

    def testVariableRankDiv(self):
        if False:
            for i in range(10):
                print('nop')
        self._VariableRankTests(state_ops.scatter_div, False)

    def testVariableRankMin(self):
        if False:
            while True:
                i = 10
        self._VariableRankTests(state_ops.scatter_min, False)

    def testVariableRankMax(self):
        if False:
            i = 10
            return i + 15
        self._VariableRankTests(state_ops.scatter_max, False)

    def testRepeatIndicesAdd(self):
        if False:
            for i in range(10):
                print('nop')
        self._VariableRankTests(state_ops.scatter_add, True)

    def testRepeatIndicesSub(self):
        if False:
            print('Hello World!')
        self._VariableRankTests(state_ops.scatter_sub, True)

    def testRepeatIndicesMul(self):
        if False:
            print('Hello World!')
        self._VariableRankTests(state_ops.scatter_mul, True)

    def testRepeatIndicesDiv(self):
        if False:
            while True:
                i = 10
        self._VariableRankTests(state_ops.scatter_div, True)

    def testRepeatIndicesMin(self):
        if False:
            i = 10
            return i + 15
        self._VariableRankTests(state_ops.scatter_min, True)

    def testRepeatIndicesMax(self):
        if False:
            return 10
        self._VariableRankTests(state_ops.scatter_max, True)

    def testVariableRankUpdateScalar(self):
        if False:
            for i in range(10):
                print('nop')
        self._VariableRankTests(state_ops.scatter_update, False, True)

    def testVariableRankAddScalar(self):
        if False:
            i = 10
            return i + 15
        self._VariableRankTests(state_ops.scatter_add, False, True)

    def testVariableRankSubScalar(self):
        if False:
            for i in range(10):
                print('nop')
        self._VariableRankTests(state_ops.scatter_sub, False, True)

    def testVariableRankMulScalar(self):
        if False:
            for i in range(10):
                print('nop')
        self._VariableRankTests(state_ops.scatter_mul, False, True)

    def testVariableRankDivScalar(self):
        if False:
            return 10
        self._VariableRankTests(state_ops.scatter_div, False, True)

    def testVariableRankMinScalar(self):
        if False:
            i = 10
            return i + 15
        self._VariableRankTests(state_ops.scatter_min, False, True)

    def testVariableRankMaxScalar(self):
        if False:
            print('Hello World!')
        self._VariableRankTests(state_ops.scatter_max, False, True)

    def testRepeatIndicesAddScalar(self):
        if False:
            i = 10
            return i + 15
        self._VariableRankTests(state_ops.scatter_add, True, True)

    def testRepeatIndicesSubScalar(self):
        if False:
            for i in range(10):
                print('nop')
        self._VariableRankTests(state_ops.scatter_sub, True, True)

    def testRepeatIndicesMulScalar(self):
        if False:
            print('Hello World!')
        self._VariableRankTests(state_ops.scatter_mul, True, True)

    def testRepeatIndicesDivScalar(self):
        if False:
            print('Hello World!')
        self._VariableRankTests(state_ops.scatter_div, True, True)

    def testRepeatIndicesMinScalar(self):
        if False:
            return 10
        self._VariableRankTests(state_ops.scatter_min, True, True)

    def testRepeatIndicesMaxScalar(self):
        if False:
            return 10
        self._VariableRankTests(state_ops.scatter_max, True, True)

    def testBooleanScatterUpdate(self):
        if False:
            while True:
                i = 10
        if not test.is_gpu_available():
            with self.session(use_gpu=False):
                var = variables.Variable([True, False])
                update0 = state_ops.scatter_update(var, 1, True)
                update1 = state_ops.scatter_update(var, constant_op.constant(0, dtype=dtypes.int64), False)
                self.evaluate(var.initializer)
                self.evaluate([update0, update1])
                self.assertAllEqual([False, True], self.evaluate(var))

    def testScatterOutOfRangeCpu(self):
        if False:
            print('Hello World!')
        for (op, _) in _TF_OPS_TO_NUMPY.items():
            params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
            updates = np.array([-3, -4, -5]).astype(np.float32)
            if not test.is_gpu_available():
                with self.session(use_gpu=False):
                    ref = variables.Variable(params)
                    self.evaluate(ref.initializer)
                    indices = np.array([2, 0, 5])
                    self.evaluate(op(ref, indices, updates))
                    indices = np.array([-1, 0, 5])
                    with self.assertRaisesOpError('indices\\[0\\] = -1 is not in \\[0, 6\\)'):
                        self.evaluate(op(ref, indices, updates))
                    indices = np.array([2, 0, 6])
                    with self.assertRaisesOpError('indices\\[2\\] = 6 is not in \\[0, 6\\)'):
                        self.evaluate(op(ref, indices, updates))

    def _disabledTestScatterOutOfRangeGpu(self):
        if False:
            i = 10
            return i + 15
        if test.is_gpu_available():
            return
        for (op, _) in _TF_OPS_TO_NUMPY.items():
            params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
            updates = np.array([-3, -4, -5]).astype(np.float32)
            with test_util.force_gpu():
                ref = variables.Variable(params)
                self.evaluate(ref.initializer)
                indices = np.array([2, 0, 5])
                self.evaluate(op(ref, indices, updates))
                indices = np.array([-1, 0, 5])
                self.evaluate(op(ref, indices, updates))
                indices = np.array([2, 0, 6])
                self.evaluate(op(ref, indices, updates))

    @test_util.run_v1_only('ResrouceVariable has deterministic scatter implementation')
    @test_util.run_cuda_only
    def testDeterminismExceptionThrowing(self):
        if False:
            for i in range(10):
                print('nop')
        v = ref_variable.RefVariable(np.array([1.0, 2.0, 3.0]))
        indices = np.array([0, 0, 0])
        updates = np.array([-3, -4, -5]).astype(np.float32)
        with test_util.deterministic_ops():
            with self.assertRaisesRegex(errors.UnimplementedError, 'Determinism is not yet supported in GPU implementation of Scatter ops'):
                self.evaluate(state_ops.scatter_update(v, indices, updates))
if __name__ == '__main__':
    test.main()