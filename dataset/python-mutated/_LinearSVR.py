from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    from sklearn.svm import LinearSVR as _LinearSVR
    import sklearn
    from . import _sklearn_util
    sklearn_class = sklearn.svm.LinearSVR
from . import _linear_regression
model_type = 'regressor'

def convert(model, features, target):
    if False:
        return 10
    'Convert a LinearSVR model to the protobuf spec.\n    Parameters\n    ----------\n    model: LinearSVR\n        A trained LinearSVR model.\n\n    feature_names: [str]\n        Name of the input columns.\n\n    target: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, _LinearSVR)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'coef_'))
    return _MLModel(_linear_regression._convert(model, features, target))

def get_input_dimension(model):
    if False:
        print('Hello World!')
    return _linear_regression.get_input_dimension(model)