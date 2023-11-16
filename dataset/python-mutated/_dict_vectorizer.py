import six as _six
from . import _sklearn_util
from ... import SPECIFICATION_VERSION
from ...models._interface_management import set_transform_interface_params
from ...proto import Model_pb2 as _Model_pb2
from ...proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from ...models._feature_management import process_or_validate_features
from ...models.feature_vectorizer import create_feature_vectorizer
from ...models import MLModel as _MLModel
from ..._deps import _HAS_SKLEARN
if _HAS_SKLEARN:
    from sklearn.feature_extraction import DictVectorizer
    sklearn_class = DictVectorizer
from ...models import datatypes
from ...models.pipeline import Pipeline
model_type = 'transformer'

def convert(model, input_features, output_features):
    if False:
        print('Hello World!')
    'Convert a _imputer model to the protobuf spec.\n\n    Parameters\n    ----------\n    model: Imputer\n        A trained Imputer model.\n\n    input_features: str\n        Name of the input column.\n\n    output_features: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    _INTERMEDIATE_FEATURE_NAME = '__sparse_vector_features__'
    n_dimensions = len(model.feature_names_)
    input_features = process_or_validate_features(input_features)
    output_features = process_or_validate_features(output_features, n_dimensions)
    pline = Pipeline(input_features, output_features)
    dv_spec = _Model_pb2.Model()
    dv_spec.specificationVersion = SPECIFICATION_VERSION
    tr_spec = dv_spec.dictVectorizer
    is_str = None
    for feature_name in model.feature_names_:
        if isinstance(feature_name, _six.string_types):
            if is_str == False:
                raise ValueError('Mapping of DictVectorizer mixes int and str types.')
            tr_spec.stringToIndex.vector.append(feature_name)
            is_str == True
        if isinstance(feature_name, _six.integer_types):
            if is_str == True:
                raise ValueError('Mapping of DictVectorizer mixes int and str types.')
            tr_spec.int64ToIndex.vector.append(feature_name)
            is_str == False
    intermediate_features = [(_INTERMEDIATE_FEATURE_NAME, datatypes.Dictionary(key_type=int))]
    set_transform_interface_params(dv_spec, input_features, intermediate_features)
    pline.add_model(dv_spec)
    (fvec, _num_out_dim) = create_feature_vectorizer(intermediate_features, output_features[0][0], {'__sparse_vector_features__': n_dimensions})
    pline.add_model(fvec)
    return _MLModel(pline.spec)

def update_dimension(m, current_num_dimensions):
    if False:
        for i in range(10):
            print('nop')
    return len(m.feature_names_)

def get_input_dimension(m):
    if False:
        print('Hello World!')
    return None

def get_input_feature_names(m):
    if False:
        i = 10
        return i + 15
    return m.feature_names_