from ...proto import Model_pb2 as _Model_pb2
from ...models._interface_management import set_regressor_interface_params
from ... import SPECIFICATION_VERSION
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    from ._sklearn_util import check_fitted
    from sklearn.svm import SVR as _SVR
    from . import _sklearn_util
    sklearn_class = _SVR
model_type = 'regressor'
from ._svm_common import _set_kernel

def _generate_base_svm_regression_spec(model):
    if False:
        i = 10
        return i + 15
    '\n    Takes an SVM regression model  produces a starting spec using the parts.\n    that are shared between all SVMs.\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    spec = _Model_pb2.Model()
    spec.specificationVersion = SPECIFICATION_VERSION
    svm = spec.supportVectorRegressor
    _set_kernel(model, svm)
    svm.rho = -model.intercept_[0]
    for i in range(len(model._dual_coef_)):
        for cur_alpha in model._dual_coef_[i]:
            svm.coefficients.alpha.append(cur_alpha)
    for cur_src_vector in model.support_vectors_:
        cur_dest_vector = svm.denseSupportVectors.vectors.add()
        for i in cur_src_vector:
            cur_dest_vector.values.append(i)
    return spec

def convert(model, features, target):
    if False:
        for i in range(10):
            print('nop')
    'Convert a Support Vector Regressor (SVR) model to the protobuf spec.\n    Parameters\n    ----------\n    model: SVR\n        A trained SVR encoder model.\n\n    feature_names: [str]\n        Name of the input columns.\n\n    target: str\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    spec = _generate_base_svm_regression_spec(model)
    spec = set_regressor_interface_params(spec, features, target)
    return _MLModel(spec)

def get_input_dimension(model):
    if False:
        return 10
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    check_fitted(model, lambda m: hasattr(m, 'support_vectors_'))
    return len(model.support_vectors_[0])