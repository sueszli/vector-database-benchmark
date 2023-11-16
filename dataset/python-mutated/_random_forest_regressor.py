from ._tree_ensemble import convert_tree_ensemble as _convert_tree_ensemble
from ._tree_ensemble import get_input_dimension
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    import sklearn.ensemble as _ensemble
    from . import _sklearn_util
    sklearn_class = _ensemble.RandomForestRegressor
model_type = 'regressor'

def convert(model, feature_names, target):
    if False:
        while True:
            i = 10
    'Convert a boosted tree model to protobuf format.\n\n    Parameters\n    ----------\n    decision_tree : RandomForestRegressor\n        A trained scikit-learn tree model.\n\n    feature_names: [str]\n        Name of the input columns.\n\n    target: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, _ensemble.RandomForestRegressor)

    def is_rf_model(m):
        if False:
            return 10
        if len(m.estimators_) == 0:
            return False
        if hasattr(m, 'estimators_') and m.estimators_ is not None:
            for t in m.estimators_:
                if not hasattr(t, 'tree_') or t.tree_ is None:
                    return False
            return True
        else:
            return False
    _sklearn_util.check_fitted(model, is_rf_model)
    return _MLModel(_convert_tree_ensemble(model, feature_names, target))