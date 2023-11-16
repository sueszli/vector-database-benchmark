from __future__ import division
import six
import chainer
from chainer import backend
from chainer import function
from chainer.utils import type_check

def _fbeta_score(precision, recall, beta):
    if False:
        for i in range(10):
            print('nop')
    beta_square = beta * beta
    return ((1 + beta_square) * precision * recall / (beta_square * precision + recall)).astype(precision.dtype, copy=False)

class ClassificationSummary(function.Function):

    def __init__(self, label_num, beta, ignore_label):
        if False:
            print('Hello World!')
        self.label_num = label_num
        self.beta = beta
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check._argname(in_types, ('x', 't'))
        (x_type, t_type) = in_types
        type_check.expect(x_type.dtype.kind == 'f', t_type.dtype.kind == 'i')
        t_ndim = type_check.eval(t_type.ndim)
        type_check.expect(x_type.ndim >= t_type.ndim, x_type.shape[0] == t_type.shape[0], x_type.shape[2:t_ndim + 1] == t_type.shape[1:])
        for i in six.moves.range(t_ndim + 1, type_check.eval(x_type.ndim)):
            type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        xp = backend.get_array_module(*inputs)
        (y, t) = inputs
        t = t.astype(xp.int32, copy=False)
        if self.label_num is None:
            label_num = xp.amax(t) + 1
        else:
            label_num = self.label_num
            if chainer.is_debug():
                assert (t < label_num).all()
        mask = (t == self.ignore_label).ravel()
        pred = xp.where(mask, label_num, y.argmax(axis=1).ravel())
        true = xp.where(mask, label_num, t.ravel())
        support = xp.bincount(true, minlength=label_num + 1)[:label_num]
        relevant = xp.bincount(pred, minlength=label_num + 1)[:label_num]
        tp_mask = xp.where(pred == true, true, label_num)
        tp = xp.bincount(tp_mask, minlength=label_num + 1)[:label_num]
        precision = tp / relevant
        recall = tp / support
        fbeta = _fbeta_score(precision, recall, self.beta)
        return (precision, recall, fbeta, support)

def classification_summary(y, t, label_num=None, beta=1.0, ignore_label=-1):
    if False:
        while True:
            i = 10
    'Calculates Precision, Recall, F beta Score, and support.\n\n    This function calculates the following quantities for each class.\n\n    - Precision: :math:`\\frac{\\mathrm{tp}}{\\mathrm{tp} + \\mathrm{fp}}`\n    - Recall: :math:`\\frac{\\mathrm{tp}}{\\mathrm{tp} + \\mathrm{fn}}`\n    - F beta Score: The weighted harmonic average of Precision and Recall.\n    - Support: The number of instances of each ground truth label.\n\n    Here, ``tp``, ``fp``, ``tn``, and ``fn`` stand for the number of true\n    positives, false positives, true negatives, and false negatives,\n    respectively.\n\n    ``label_num`` specifies the number of classes, that is,\n    each value in ``t`` must be an integer in the range of\n    ``[0, label_num)``.\n    If ``label_num`` is ``None``, this function regards\n    ``label_num`` as a maximum of in ``t`` plus one.\n\n    ``ignore_label`` determines which instances should be ignored.\n    Specifically, instances with the given label are not taken\n    into account for calculating the above quantities.\n    By default, it is set to -1 so that all instances are taken\n    into consideration, as labels are supposed to be non-negative integers.\n    Setting ``ignore_label`` to a non-negative integer less than ``label_num``\n    is illegal and yields undefined behavior. In the current implementation,\n    it arises ``RuntimeWarning`` and ``ignore_label``-th entries in output\n    arrays do not contain correct quantities.\n\n    Args:\n        y (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Variable holding a vector of scores.\n        t (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Variable holding a vector of ground truth labels.\n        label_num (int): The number of classes.\n        beta (float): The parameter which determines the weight of\n            precision in the F-beta score.\n        ignore_label (int): Instances with this label are ignored.\n\n    Returns:\n        4-tuple of ~chainer.Variable of size ``(label_num,)``.\n        Each element represents precision, recall, F beta score,\n        and support of this minibatch.\n\n    '
    return ClassificationSummary(label_num, beta, ignore_label)(y, t)

def precision(y, t, label_num=None, ignore_label=-1):
    if False:
        while True:
            i = 10
    ret = ClassificationSummary(label_num, 1.0, ignore_label)(y, t)
    return (ret[0], ret[-1])

def recall(y, t, label_num=None, ignore_label=-1):
    if False:
        for i in range(10):
            print('nop')
    ret = ClassificationSummary(label_num, 1.0, ignore_label)(y, t)
    return (ret[1], ret[-1])

def fbeta_score(y, t, label_num=None, beta=1.0, ignore_label=-1):
    if False:
        for i in range(10):
            print('nop')
    ret = ClassificationSummary(label_num, beta, ignore_label)(y, t)
    return (ret[2], ret[-1])

def f1_score(y, t, label_num=None, ignore_label=-1):
    if False:
        for i in range(10):
            print('nop')
    ret = ClassificationSummary(label_num, 1.0, ignore_label)(y, t)
    return (ret[2], ret[-1])