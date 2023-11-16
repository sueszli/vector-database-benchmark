from ... import SPECIFICATION_VERSION
from ...models._interface_management import set_regressor_interface_params
from ...proto import Model_pb2 as _Model_pb2
from ...proto import FeatureTypes_pb2 as _FeatureTypes_pb2
import numpy as _np
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    from . import _sklearn_util
    import sklearn
    from sklearn.linear_model import LinearRegression
    model_type = 'regressor'
    sklearn_class = sklearn.linear_model.LinearRegression

def convert(model, features, target):
    if False:
        return 10
    'Convert a linear regression model to the protobuf spec.\n    Parameters\n    ----------\n    model: LinearRegression\n        A trained linear regression encoder model.\n\n    feature_names: [str]\n        Name of the input columns.\n\n    target: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, LinearRegression)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'coef_'))
    return _MLModel(_convert(model, features, target))

def _convert(model, features, target):
    if False:
        for i in range(10):
            print('nop')
    spec = _Model_pb2.Model()
    spec.specificationVersion = SPECIFICATION_VERSION
    spec = set_regressor_interface_params(spec, features, target)
    lr = spec.glmRegressor
    if isinstance(model.intercept_, _np.ndarray):
        assert len(model.intercept_) == 1
        lr.offset.append(model.intercept_[0])
    else:
        lr.offset.append(model.intercept_)
    weights = lr.weights.add()
    for i in model.coef_:
        weights.value.append(i)
    return spec

def get_input_dimension(model):
    if False:
        return 10
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'coef_'))
    return model.coef_.size