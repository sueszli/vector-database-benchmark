from ...proto import Model_pb2 as _Model_pb2
from ...proto import SVM_pb2 as _SVM_pb2
from ... import SPECIFICATION_VERSION as _SPECIFICATION_VERSION
from ...models._interface_management import set_classifier_interface_params
from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
if _HAS_SKLEARN:
    from ._sklearn_util import check_fitted
    from sklearn.svm import SVC as _SVC
    sklearn_class = _SVC
model_type = 'classifier'
from ._svm_common import _set_kernel

def _generate_base_svm_classifier_spec(model):
    if False:
        return 10
    '\n    Takes an SVM classifier produces a starting spec using the parts.  that are\n    shared between all SVMs.\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    check_fitted(model, lambda m: hasattr(m, 'support_vectors_'))
    spec = _Model_pb2.Model()
    spec.specificationVersion = _SPECIFICATION_VERSION
    svm = spec.supportVectorClassifier
    _set_kernel(model, svm)
    for cur_rho in model.intercept_:
        if len(model.classes_) == 2:
            svm.rho.append(cur_rho)
        else:
            svm.rho.append(-cur_rho)
    for i in range(len(model._dual_coef_)):
        svm.coefficients.add()
        for cur_alpha in model._dual_coef_[i]:
            svm.coefficients[i].alpha.append(cur_alpha)
    for cur_src_vector in model.support_vectors_:
        cur_dest_vector = svm.denseSupportVectors.vectors.add()
        for i in cur_src_vector:
            cur_dest_vector.values.append(i)
    return spec

def convert(model, feature_names, target):
    if False:
        while True:
            i = 10
    'Convert a Support Vector Classtion (SVC) model to the protobuf spec.\n    Parameters\n    ----------\n    model: SVC\n        A trained SVC encoder model.\n\n    feature_names: [str], optional (default=None)\n        Name of the input columns.\n\n    target: str, optional (default=None)\n        Name of the output column.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    spec = _generate_base_svm_classifier_spec(model)
    spec = set_classifier_interface_params(spec, feature_names, model.classes_, 'supportVectorClassifier', output_features=target)
    svm = spec.supportVectorClassifier
    for i in model.n_support_:
        svm.numberOfSupportVectorsPerClass.append(int(i))
    if len(model.probA_) != 0 and len(model.classes_) == 2:
        print('[WARNING] Scikit Learn uses a technique to normalize pairwise probabilities even for binary classification. This can cause differences in predicted probabilities, usually less than 0.5%.')
    if len(model.probA_) != 0:
        for i in model.probA_:
            svm.probA.append(i)
    for i in model.probB_:
        svm.probB.append(i)
    return _MLModel(spec)

def supports_output_scores(model):
    if False:
        while True:
            i = 10
    return len(model.probA_) != 0

def get_output_classes(model):
    if False:
        return 10
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    check_fitted(model, lambda m: hasattr(m, 'support_vectors_'))
    return list(model.classes_)

def get_input_dimension(model):
    if False:
        i = 10
        return i + 15
    if not _HAS_SKLEARN:
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    check_fitted(model, lambda m: hasattr(m, 'support_vectors_'))
    return len(model.support_vectors_[0])