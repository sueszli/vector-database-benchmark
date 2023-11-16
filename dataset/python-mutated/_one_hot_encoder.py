from . import _sklearn_util
from ... import SPECIFICATION_VERSION
from ...models._interface_management import set_transform_interface_params
from ...proto import Model_pb2 as _Model_pb2
from ...proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from ...proto import OneHotEncoder_pb2 as _OHE_pb2
from ...models import datatypes
from ...models import MLModel as _MLModel
from ...models.feature_vectorizer import create_feature_vectorizer
from ...models.array_feature_extractor import create_array_feature_extractor
from ...models.pipeline import Pipeline
from ..._deps import _HAS_SKLEARN as _HAS_SKLEARN
if _HAS_SKLEARN:
    import sklearn
    from sklearn.preprocessing import OneHotEncoder
    sklearn_class = OneHotEncoder
model_type = 'transformer'

def convert(model, input_features, output_features):
    if False:
        while True:
            i = 10
    'Convert a one-hot-encoder model to the protobuf spec.\n\n    Parameters\n    ----------\n    model: OneHotEncoder\n        A trained one-hot encoder model.\n\n    input_features: str, optional\n        Name of the input column.\n\n    output_features: str, optional\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, OneHotEncoder)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'active_features_'))
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'n_values_'))
    input_dimension = get_input_dimension(model)
    if input_dimension is not None:
        assert len(input_features) == 1
        assert input_features[0][1] == datatypes.Array(input_dimension)
    input_dimension = input_features[0][1].num_elements
    expected_output_dimension = update_dimension(model, input_dimension)
    assert output_features[0][1] == datatypes.Array(expected_output_dimension)
    feature_vectorizer_input_features = []
    feature_vectorizer_size_map = {}
    if model.categorical_features == 'all':
        _categorical_features = set(range(input_dimension))
        _cat_feature_idx_mapping = dict(((i, i) for i in range(input_dimension)))
    else:
        _categorical_features = set(model.categorical_features)
        _cat_feature_idx_mapping = dict(((_idx, i) for (i, _idx) in enumerate(sorted(model.categorical_features))))
    pline = Pipeline(input_features, output_features)
    pack_idx = 0
    for idx in range(input_dimension):
        f_name = '__OHE_%d__' % pack_idx
        if idx in _categorical_features:
            feature_extractor_spec = create_array_feature_extractor(input_features, f_name, idx, output_type='Int64')
            pline.add_model(feature_extractor_spec)
            _cat_feature_idx = _cat_feature_idx_mapping[idx]
            ohe_input_features = [(f_name, datatypes.Int64())]
            ohe_output_features = [(f_name, datatypes.Dictionary('Int64'))]
            o_spec = _Model_pb2.Model()
            o_spec.specificationVersion = SPECIFICATION_VERSION
            o_spec = set_transform_interface_params(o_spec, ohe_input_features, ohe_output_features)
            ohe_spec = o_spec.oneHotEncoder
            ohe_spec.outputSparse = True
            if model.handle_unknown == 'error':
                ohe_spec.handleUnknown = _OHE_pb2.OneHotEncoder.HandleUnknown.Value('ErrorOnUnknown')
            else:
                ohe_spec.handleUnknown = _OHE_pb2.OneHotEncoder.HandleUnknown.Value('IgnoreUnknown')

            def bs_find(a, i):
                if False:
                    print('Hello World!')
                (lb, k) = (0, len(a))
                while k > 0:
                    _idx = lb + k // 2
                    if a[_idx] < i:
                        lb = _idx + 1
                        k -= 1
                    k = k // 2
                return lb
            f_idx_bottom = model.feature_indices_[_cat_feature_idx]
            f_idx_top = model.feature_indices_[_cat_feature_idx + 1]
            cat_feat_idx_bottom = bs_find(model.active_features_, f_idx_bottom)
            cat_feat_idx_top = bs_find(model.active_features_, f_idx_top)
            n_cat_values = cat_feat_idx_top - cat_feat_idx_bottom
            for i in range(cat_feat_idx_bottom, cat_feat_idx_top):
                cat_idx = model.active_features_[i] - f_idx_bottom
                ohe_spec.int64Categories.vector.append(cat_idx)
            pline.add_model(o_spec)
            feature_vectorizer_input_features.append((f_name, datatypes.Dictionary('Int64')))
            feature_vectorizer_size_map[f_name] = n_cat_values
            pack_idx += 1
    pass_through_features = [idx for idx in range(input_dimension) if idx not in _categorical_features]
    if pass_through_features:
        f_name = '__OHE_pass_through__'
        feature_extractor_spec = create_array_feature_extractor(input_features, f_name, pass_through_features)
        pline.add_model(feature_extractor_spec)
        feature_vectorizer_input_features.append((f_name, datatypes.Array(len(pass_through_features))))
    output_feature_name = output_features[0][0]
    output_feature_dimension = output_features[0][1].num_elements
    (fvec, _num_out_dim) = create_feature_vectorizer(feature_vectorizer_input_features, output_features[0][0], feature_vectorizer_size_map)
    assert _num_out_dim == output_features[0][1].num_elements
    pline.add_model(fvec)
    return _MLModel(pline.spec)

def update_dimension(model, input_dimension):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a model that takes an array of dimension input_dimension, returns\n    the output dimension.\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'active_features_'))
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'n_values_'))
    if model.categorical_features == 'all':
        return len(model.active_features_)
    else:
        out_dimension = len(model.active_features_) + (input_dimension - len(model.n_values_))
    return out_dimension

def get_input_dimension(model):
    if False:
        while True:
            i = 10
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'active_features_'))
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'n_values_'))
    if model.categorical_features == 'all':
        return len(model.feature_indices_) - 1
    else:
        return None