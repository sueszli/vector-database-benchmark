from chainer import backend
from chainer import function
from chainer.utils import type_check

class R2_score(function.Function):

    def __init__(self, sample_weight, multioutput):
        if False:
            return 10
        if sample_weight is not None:
            raise NotImplementedError()
        if multioutput in ['uniform_average', 'raw_values']:
            self.multioutput = multioutput
        else:
            raise ValueError('invalid multioutput argument')

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check._argname(in_types, ('pred', 'true'))
        (pred_type, true_type) = in_types
        type_check.expect(pred_type.dtype.kind == 'f', true_type.dtype.kind == 'f')
        type_check.expect(pred_type.shape == true_type.shape)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        xp = backend.get_array_module(*inputs)
        (pred, true) = inputs
        SS_res = xp.asarray(xp.sum((pred - true) ** 2, axis=0))
        SS_tot = xp.asarray(xp.sum((true - xp.mean(true, axis=0)) ** 2, axis=0))
        SS_tot_iszero = SS_tot == 0
        SS_tot[SS_tot_iszero] = 1
        ret = xp.where(SS_tot_iszero, 0.0, 1 - SS_res / SS_tot).astype(pred.dtype, copy=False)
        if self.multioutput == 'uniform_average':
            return (xp.asarray(ret.mean()),)
        elif self.multioutput == 'raw_values':
            return (ret,)

def r2_score(pred, true, sample_weight=None, multioutput='uniform_average'):
    if False:
        return 10
    "Computes R^2(coefficient of determination) regression score function.\n\n    Args:\n        pred (:class:`~chainer.Variable` or :ref:`ndarray`): Variable holding a\n            vector, matrix or tensor of estimated target values.\n        true (:class:`~chainer.Variable` or :ref:`ndarray`): Variable holding a\n            vector, matrix or tensor of correct target values.\n        sample_weight: This argument is for compatibility with scikit-learn's\n                implementation of r2_score. Current implementation admits None\n                only.\n        multioutput(string): ['uniform_average', 'raw_values']. if\n                'uniform_average', this function returns an average of R^2\n                score of multiple output. If 'raw_average', this function\n                return a set of R^2 score of multiple output.\n    Returns:\n        ~chainer.Variable: A Variable holding a scalar array of the R^2 score\n        if 'multioutput' is 'uniform_average' or a vector of R^2 scores if\n        'multioutput' is 'raw_values'.\n\n    .. note:: This function is non-differentiable.\n\n    "
    return R2_score(sample_weight=sample_weight, multioutput=multioutput)(pred, true)