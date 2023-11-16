from . import _SVR as _SVR
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    from ._sklearn_util import check_fitted
    from sklearn.svm import NuSVR as _NuSVR
    from . import _sklearn_util
    sklearn_class = _NuSVR
model_type = 'regressor'

def convert(model, feature_names, target):
    if False:
        return 10
    'Convert a Nu Support Vector Regression (NuSVR) model to the protobuf spec.\n    Parameters\n    ----------\n    model: NuSVR\n        A trained NuSVR encoder model.\n\n    feature_names: [str]\n        Name of the input columns.\n\n    target: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, _NuSVR)
    return _SVR.convert(model, feature_names, target)

def get_input_dimension(model):
    if False:
        for i in range(10):
            print('nop')
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    check_fitted(model, lambda m: hasattr(m, 'support_vectors_'))
    return _SVR.get_input_dimension(model)