from ._tree_ensemble import convert_tree_ensemble as _convert_tree_ensemble
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    import sklearn.tree as _tree
    from . import _sklearn_util
model_type = 'regressor'
sklearn_class = _tree.DecisionTreeRegressor

def convert(model, feature_names, target):
    if False:
        print('Hello World!')
    'Convert a decision tree model to protobuf format.\n\n    Parameters\n    ----------\n    decision_tree : DecisionTreeRegressor\n        A trained scikit-learn tree model.\n\n    feature_names: [str]\n        Name of the input columns.\n\n    target: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, _tree.DecisionTreeRegressor)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'tree_') and model.tree_ is not None)
    return _MLModel(_convert_tree_ensemble(model, feature_names, target))

def get_input_dimension(model):
    if False:
        print('Hello World!')
    return model.n_features_