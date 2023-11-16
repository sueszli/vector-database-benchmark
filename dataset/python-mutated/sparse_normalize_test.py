import caffe2.python.hypothesis_test_util as hu
import hypothesis
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core
from hypothesis import HealthCheck, given, settings

class TestSparseNormalize(hu.HypothesisTestCase):

    @staticmethod
    def ref_normalize(param_in, use_max_norm, norm):
        if False:
            i = 10
            return i + 15
        param_norm = np.linalg.norm(param_in) + 1e-12
        if use_max_norm and param_norm > norm or not use_max_norm:
            param_in = param_in * norm / param_norm
        return param_in

    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(inputs=hu.tensors(n=2, min_dim=2, max_dim=2), use_max_norm=st.booleans(), norm=st.floats(min_value=1.0, max_value=4.0), data_strategy=st.data(), use_fp16=st.booleans(), **hu.gcs_cpu_only)
    def test_sparse_normalize(self, inputs, use_max_norm, norm, data_strategy, use_fp16, gc, dc):
        if False:
            print('Hello World!')
        (param, grad) = inputs
        param += 0.02 * np.sign(param)
        param[param == 0.0] += 0.02
        if use_fp16:
            param = param.astype(np.float16)
            grad = grad.astype(np.float16)
        indices = data_strategy.draw(hu.tensor(dtype=np.int64, min_dim=1, max_dim=1, elements=st.sampled_from(np.arange(param.shape[0]))))
        hypothesis.note('indices.shape: %s' % str(indices.shape))
        hypothesis.assume(np.array_equal(np.unique(indices.flatten()), np.sort(indices.flatten())))
        op1 = core.CreateOperator('Float16SparseNormalize' if use_fp16 else 'SparseNormalize', ['param', 'indices'], ['param'], use_max_norm=use_max_norm, norm=norm)
        grad = grad[indices]
        op2 = core.CreateOperator('Float16SparseNormalize' if use_fp16 else 'SparseNormalize', ['param', 'indices', 'grad'], ['param'], use_max_norm=use_max_norm, norm=norm)

        def ref_sparse_normalize(param, indices, grad=None):
            if False:
                print('Hello World!')
            param_out = np.copy(param)
            for (_, index) in enumerate(indices):
                param_out[index] = self.ref_normalize(param[index], use_max_norm, norm)
            return (param_out,)
        self.assertReferenceChecks(gc, op1, [param, indices], ref_sparse_normalize, threshold=0.01 if use_fp16 else 0.0001)
        self.assertReferenceChecks(gc, op2, [param, indices, grad], ref_sparse_normalize, threshold=0.01 if use_fp16 else 0.0001)