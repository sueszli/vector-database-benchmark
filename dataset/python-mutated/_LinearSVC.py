from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    from sklearn.svm import LinearSVC as _LinearSVC
    sklearn_class = _LinearSVC
    from . import _sklearn_util
from . import _logistic_regression
model_type = 'classifier'

def convert(model, feature_names, target):
    if False:
        return 10
    'Convert a LinearSVC model to the protobuf spec.\n    Parameters\n    ----------\n    model: LinearSVC\n        A trained LinearSVC model.\n\n    feature_names: [str]\n        Name of the input columns.\n\n    target: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, _LinearSVC)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'coef_'))
    return _MLModel(_logistic_regression._convert(model, feature_names, target))

def supports_output_scores(model):
    if False:
        for i in range(10):
            print('nop')
    return True

def get_output_classes(model):
    if False:
        print('Hello World!')
    return _logistic_regression.get_output_classes(model)

def get_input_dimension(model):
    if False:
        while True:
            i = 10
    return _logistic_regression.get_input_dimension(model)