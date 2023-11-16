from ._tree_ensemble import convert_tree_ensemble as _convert_tree_ensemble
from ...models import MLModel as _MLModel
from coremltools import __version__ as ct_version
from coremltools.models import _METADATA_VERSION, _METADATA_SOURCE

def convert(model, feature_names=None, target='target', force_32bit_float=True, mode='regressor', class_labels=None, n_classes=None):
    if False:
        i = 10
        return i + 15
    "\n    Convert a trained XGBoost model to Core ML format.\n\n    Parameters\n    ----------\n    decision_tree : Booster\n        A trained XGboost tree model.\n\n    feature_names: [str] | str\n        Names of input features that will be exposed in the Core ML model\n        interface.\n\n        Can be set to one of the following:\n\n        - None for using the feature names from the model.\n        - List of names of the input features that should be exposed in the\n          interface to the Core ML model. These input features are in the same\n          order as the XGboost model.\n\n    target: str\n        Name of the output feature name exposed to the Core ML model.\n\n    force_32bit_float: bool\n        If True, then the resulting CoreML model will use 32 bit floats internally.\n\n    mode: str in ['regressor', 'classifier']\n        Mode of the tree model.\n\n    class_labels: list[int] or None\n        List of classes. When set to None, the class labels are just the range from\n        0 to n_classes - 1.\n\n    n_classes: int or None\n        Number of classes in classification. When set to None, the number of\n        classes is expected from the model or class_labels should be provided.\n\n    Returns\n    -------\n    model:MLModel\n        Returns an MLModel instance representing a Core ML model.\n\n    Examples\n    --------\n    .. sourcecode:: python\n\n\t\t# Convert it with default input and output names\n   \t\t>>> import coremltools\n\t\t>>> coreml_model = coremltools.converters.xgboost.convert(model)\n\n\t\t# Saving the Core ML model to a file.\n\t\t>>> coremltools.save('my_model.mlmodel')\n    "
    model = _MLModel(_convert_tree_ensemble(model, feature_names, target, force_32bit_float=force_32bit_float, mode=mode, class_labels=class_labels, n_classes=n_classes))
    from xgboost import __version__ as xgboost_version
    model.user_defined_metadata[_METADATA_VERSION] = ct_version
    model.user_defined_metadata[_METADATA_SOURCE] = 'xgboost=={0}'.format(xgboost_version)
    return model