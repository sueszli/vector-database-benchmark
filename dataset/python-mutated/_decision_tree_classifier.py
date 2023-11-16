from ._tree_ensemble import convert_tree_ensemble
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    import sklearn.tree as _tree
    from . import _sklearn_util
model_type = 'classifier'
sklearn_class = _tree.DecisionTreeClassifier

def convert(model, input_name, output_features):
    if False:
        i = 10
        return i + 15
    'Convert a decision tree model to protobuf format.\n\n    Parameters\n    ----------\n    decision_tree : DecisionTreeClassifier\n        A trained scikit-learn tree model.\n\n    input_name: str\n        Name of the input columns.\n\n    output_name: str\n        Name of the output columns.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    _sklearn_util.check_expected_type(model, _tree.DecisionTreeClassifier)
    _sklearn_util.check_fitted(model, lambda m: hasattr(m, 'tree_') and model.tree_ is not None)
    return _MLModel(convert_tree_ensemble(model, input_name, output_features, mode='classifier', class_labels=model.classes_))

def supports_output_scores(model):
    if False:
        i = 10
        return i + 15
    return True

def get_output_classes(model):
    if False:
        for i in range(10):
            print('nop')
    return list(model.classes_)

def get_input_dimension(model):
    if False:
        while True:
            i = 10
    return model.n_features_