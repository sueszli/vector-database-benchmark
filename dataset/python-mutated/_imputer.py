from . import _sklearn_util
from ... import SPECIFICATION_VERSION
from ...models._interface_management import set_transform_interface_params
from ...proto import Model_pb2 as _Model_pb2
from ...models import datatypes
from ...models import MLModel as _MLModel
from ..._deps import _HAS_SKLEARN
if _HAS_SKLEARN:
    import sklearn
    try:
        from sklearn.impute import SimpleImputer as Imputer
        sklearn_class = sklearn.impute.SimpleImputer
    except ImportError:
        from sklearn.preprocessing import Imputer
        sklearn_class = sklearn.preprocessing.Imputer
    model_type = 'transformer'

def convert(model, input_features, output_features):
    if False:
        return 10
    'Convert a DictVectorizer model to the protobuf spec.\n\n    Parameters\n    ----------\n    model: DictVectorizer\n        A fitted DictVectorizer model.\n\n    input_features: str\n        Name of the input column.\n\n    output_features: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    spec = _Model_pb2.Model()
    spec.specificationVersion = SPECIFICATION_VERSION
    assert len(input_features) == 1
    assert isinstance(input_features[0][1], datatypes.Array)
    spec = set_transform_interface_params(spec, input_features, output_features)
    _sklearn_util.check_expected_type(model, Imputer)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'statistics_'))
    if model.axis != 0:
        raise ValueError('Imputation is only supported along axis = 0.')
    tr_spec = spec.imputer
    for v in model.statistics_:
        tr_spec.imputedDoubleArray.vector.append(v)
    try:
        tr_spec.replaceDoubleValue = float(model.missing_values)
    except ValueError:
        raise ValueError('Only scalar values or NAN as missing_values in _imputer are supported.')
    return _MLModel(spec)

def update_dimension(model, input_dimension):
    if False:
        print('Hello World!')
    '\n    Given a model that takes an array of dimension input_dimension, returns\n    the output dimension.\n    '
    return input_dimension

def get_input_dimension(model):
    if False:
        return 10
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'statistics_'))
    return len(model.statistics_)