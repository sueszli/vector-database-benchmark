from . import _SVC as _SVC
from ..._deps import _HAS_SKLEARN
if _HAS_SKLEARN:
    from ._sklearn_util import check_fitted
    from . import _sklearn_util
    from sklearn.svm import NuSVC as _NuSVC
    sklearn_class = _NuSVC
model_type = 'classifier'

def convert(model, feature_names, target):
    if False:
        for i in range(10):
            print('nop')
    'Convert a Nu-Support Vector Classification (NuSVC) model to the protobuf spec.\n    Parameters\n    ----------\n    model: NuSVC\n        A trained NuSVC encoder model.\n\n    feature_names: [str], optional (default=None)\n        Name of the input columns.\n\n    target: str, optional (default=None)\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, _NuSVC)
    return _SVC.convert(model, feature_names, target)

def supports_output_scores(model):
    if False:
        i = 10
        return i + 15
    return _SVC.supports_output_scores(model)

def get_output_classes(model):
    if False:
        print('Hello World!')
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    check_fitted(model, lambda m: hasattr(m, 'support_vectors_'))
    return _SVC.get_output_classes(model)

def get_input_dimension(model):
    if False:
        i = 10
        return i + 15
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    check_fitted(model, lambda m: hasattr(m, 'support_vectors_'))
    return _SVC.get_input_dimension(model)