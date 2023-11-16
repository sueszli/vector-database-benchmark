"""
Multinomial model.

:copyright: (c) 2016 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o.utils.compatibility import *
from h2o import expr
from ..frame import H2OFrame

class TransformAttributeError(AttributeError):

    def __init__(self, obj, method):
        if False:
            i = 10
            return i + 15
        super(AttributeError, self).__init__('No {} method for {}'.format(method, obj.__class__.__name__))

class H2OTransformer(object):
    """
    H2O Transforms.

    H2O Transforms implement the following methods:

      - fit
      - transform
      - fit_transform
      - inverse_transform
      - export
      - to_rest
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def fit(self, X, y=None, **params):
        if False:
            print('Hello World!')
        raise TransformAttributeError(self, 'fit')

    def transform(self, X, y=None, **params):
        if False:
            i = 10
            return i + 15
        raise TransformAttributeError(self, 'transform')

    def inverse_transform(self, X, y=None, **params):
        if False:
            print('Hello World!')
        raise TransformAttributeError(self, 'inverse_transform')

    def export(self, X, y, **params):
        if False:
            while True:
                i = 10
        raise TransformAttributeError(self, 'export')

    def fit_transform(self, X, y=None, **params):
        if False:
            for i in range(10):
                print('nop')
        return self.fit(X, y, **params).transform(X, **params)

    def get_params(self, deep=True):
        if False:
            while True:
                i = 10
        '\n        Get parameters for this estimator.\n\n        :param bool deep: if True, return parameters of all subobjects that are estimators.\n\n        :returns: A dict of parameters.\n        '
        out = dict()
        for (key, value) in self.parms.items():
            if deep and isinstance(value, H2OTransformer):
                deep_items = list(value.get_params().items())
                out.update(((key + '__' + k, val) for (k, val) in deep_items))
            out[key] = value
        return out

    def set_params(self, **params):
        if False:
            print('Hello World!')
        self.parms.update(params)
        return self

    @staticmethod
    def _dummy_frame():
        if False:
            print('Hello World!')
        fr = H2OFrame._expr(expr.ExprNode())
        fr._is_frame = False
        fr._ex._children = None
        fr._ex._cache.dummy_fill()
        return fr

    def to_rest(self, args):
        if False:
            i = 10
            return i + 15
        return '__'.join((str(a) for a in args))