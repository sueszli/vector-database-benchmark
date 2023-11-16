from ... import SPECIFICATION_VERSION
from ...models._interface_management import set_transform_interface_params as _set_transform_interface_params
from ...proto import Model_pb2 as _Model_pb2
from ...proto.Normalizer_pb2 import Normalizer as _proto__normalizer
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    from . import _sklearn_util
    from sklearn.preprocessing import Normalizer
    sklearn_class = Normalizer
model_type = 'transformer'

def convert(model, input_features, output_features):
    if False:
        print('Hello World!')
    'Convert a normalizer model to the protobuf spec.\n\n    Parameters\n    ----------\n    model: Normalizer\n        A Normalizer.\n\n    input_features: str\n        Name of the input column.\n\n    output_features: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, Normalizer)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'norm'))
    spec = _Model_pb2.Model()
    spec.specificationVersion = SPECIFICATION_VERSION
    spec = _set_transform_interface_params(spec, input_features, output_features)
    _normalizer_spec = spec.normalizer
    if model.norm == 'l1':
        _normalizer_spec.normType = _proto__normalizer.L1
    elif model.norm == 'l2':
        _normalizer_spec.normType = _proto__normalizer.L2
    elif model.norm == 'max':
        _normalizer_spec.normType = _proto__normalizer.LMax
    return _MLModel(spec)

def update_dimension(model, input_dimension):
    if False:
        i = 10
        return i + 15
    '\n    Given a model that takes an array of dimension input_dimension, returns\n    the output dimension.\n    '
    return input_dimension

def get_input_dimension(model):
    if False:
        for i in range(10):
            print('nop')
    return None