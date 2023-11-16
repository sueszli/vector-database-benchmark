from collections import defaultdict
from time import time
try:
    from inspect import signature
except ImportError:
    from ..externals.signature_py27 import signature

class _BaseModel(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._init_time = time()

    def _check_arrays(self, X, y=None):
        if False:
            i = 10
            return i + 15
        if isinstance(X, list):
            raise ValueError('X must be a numpy array')
        if not len(X.shape) == 2:
            raise ValueError('X must be a 2D array. Try X[:, numpy.newaxis]')
        try:
            if y is None:
                return
        except AttributeError:
            if not len(y.shape) == 1:
                raise ValueError('y must be a 1D array.')
        if not len(y) == X.shape[0]:
            raise ValueError('X and y must contain the same number of samples')

    @classmethod
    def _get_param_names(cls):
        if False:
            while True:
                i = 10
        'Get parameter names for the estimator\n\n        adapted from\n        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py\n        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>\n        License: BSD 4 clause\n\n\n\n        '
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []
        init_signature = signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always specify their parameters in the signature of their __init__ (no varargs). %s with constructor %s doesn't  follow this convention." % (cls, init_signature))
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        if False:
            return 10
        "Get parameters for this estimator.\n\n        Parameters\n        ----------\n        deep : boolean, optional\n            If True, will return the parameters for this estimator and\n            contained subobjects that are estimators.\n\n        Returns\n        -------\n        params : mapping of string to any\n            Parameter names mapped to their values.'\n\n        adapted from\n        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py\n        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>\n        License: BSD 3 clause\n\n        "
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update(((key + '__' + k, val) for (k, val) in deep_items))
            out[key] = value
        return out

    def set_params(self, **params):
        if False:
            for i in range(10):
                print('nop')
        "Set the parameters of this estimator.\n        The method works on simple estimators as well as on nested objects\n        (such as pipelines). The latter have parameters of the form\n        ``<component>__<parameter>`` so that it's possible to update each\n        component of a nested object.\n\n        Returns\n        -------\n        self\n\n        adapted from\n        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py\n        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>\n        License: BSD 3 clause\n\n        "
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        nested_params = defaultdict(dict)
        for (key, value) in params.items():
            (key, delim, sub_key) = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. Check the list of available parameters with `estimator.get_params().keys()`.' % (key, self))
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for (key, sub_params) in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self