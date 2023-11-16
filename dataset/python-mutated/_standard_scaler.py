from ... import SPECIFICATION_VERSION
from ...models._interface_management import set_transform_interface_params as _set_transform_interface_params
from ...proto import Model_pb2 as _Model_pb2
from ...proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    from . import _sklearn_util
    import sklearn
    from sklearn.preprocessing import StandardScaler
    sklearn_class = StandardScaler
model_type = 'transformer'

def convert(model, input_features, output_features):
    if False:
        return 10
    'Convert a _imputer model to the protobuf spec.\n\n    Parameters\n    ----------\n    model: Imputer\n        A trained Imputer model.\n\n    input_features: str\n        Name of the input column.\n\n    output_features: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, StandardScaler)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'mean_'))
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'scale_'))
    spec = _Model_pb2.Model()
    spec.specificationVersion = SPECIFICATION_VERSION
    spec = _set_transform_interface_params(spec, input_features, output_features)
    tr_spec = spec.scaler
    for x in model.mean_:
        tr_spec.shiftValue.append(-x)
    for x in model.scale_:
        tr_spec.scaleValue.append(1.0 / x)
    return _MLModel(spec)

def update_dimension(model, input_dimension):
    if False:
        print('Hello World!')
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'mean_'))
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'scale_'))
    return input_dimension

def get_input_dimension(model):
    if False:
        return 10
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'mean_'))
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'scale_'))
    return len(model.mean_)