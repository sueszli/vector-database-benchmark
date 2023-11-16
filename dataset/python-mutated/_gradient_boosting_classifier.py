from ._tree_ensemble import convert_tree_ensemble as _convert_tree_ensemble
from ._tree_ensemble import get_input_dimension
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    import sklearn.ensemble as _ensemble
    from . import _sklearn_util
    sklearn_class = _ensemble.GradientBoostingClassifier
model_type = 'classifier'

def convert(model, feature_names, target):
    if False:
        print('Hello World!')
    'Convert a boosted tree model to protobuf format.\n\n    Parameters\n    ----------\n    decision_tree : GradientBoostingClassifier\n        A trained scikit-learn tree model.\n\n    feature_names: [str]\n        Name of the input columns.\n\n    target: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, _ensemble.GradientBoostingClassifier)

    def is_gbr_model(m):
        if False:
            return 10
        if len(m.estimators_) == 0:
            return False
        if hasattr(m, 'estimators_') and m.estimators_ is not None:
            for t in m.estimators_.flatten():
                if not hasattr(t, 'tree_') or t.tree_ is None:
                    return False
            return True
        else:
            return False
    _sklearn_util.check_fitted(model, is_gbr_model)
    post_evaluation_transform = None
    if model.n_classes_ == 2:
        base_prediction = [model.init_.prior]
        post_evaluation_transform = 'Regression_Logistic'
    else:
        base_prediction = list(model.init_.priors)
        post_evaluation_transform = 'Classification_SoftMax'
    return _MLModel(_convert_tree_ensemble(model, feature_names, target, mode='classifier', base_prediction=base_prediction, class_labels=model.classes_, post_evaluation_transform=post_evaluation_transform))

def supports_output_scores(model):
    if False:
        return 10
    return True

def get_output_classes(model):
    if False:
        while True:
            i = 10
    return list(model.classes_)