from ... import SPECIFICATION_VERSION
from ..._deps import _HAS_LIBSVM
from coremltools import __version__ as ct_version
from coremltools.models import _METADATA_VERSION, _METADATA_SOURCE
from six import string_types as _string_types

def _infer_min_num_features(model):
    if False:
        for i in range(10):
            print('nop')
    max_index = 0
    for i in range(model.l):
        j = 0
        while model.SV[i][j].index != -1:
            cur_last_index = model.SV[i][j].index
            j += 1
        if cur_last_index > max_index:
            max_index = cur_last_index
    return max_index

def convert(libsvm_model, feature_names, target, input_length, probability):
    if False:
        i = 10
        return i + 15
    'Convert a svm model to the protobuf spec.\n\n    This currently supports:\n      * C-SVC\n      * nu-SVC\n      * Epsilon-SVR\n      * nu-SVR\n\n    Parameters\n    ----------\n    model_path: libsvm_model\n       Libsvm representation of the model.\n\n    feature_names : [str] | str\n        Names of each of the features.\n\n    target: str\n        Name of the predicted class column.\n\n    probability: str\n        Name of the class probability column. Only used for C-SVC and nu-SVC.\n\n    Returns\n    -------\n    model_spec: An object of type Model_pb.\n        Protobuf representation of the model\n    '
    if not _HAS_LIBSVM:
        raise RuntimeError('libsvm not found. libsvm conversion API is disabled.')
    from libsvm import svm as _svm
    from ...proto import SVM_pb2
    from ...proto import Model_pb2
    from ...proto import FeatureTypes_pb2
    from ...models import MLModel
    svm_type_enum = libsvm_model.param.svm_type
    export_spec = Model_pb2.Model()
    export_spec.specificationVersion = SPECIFICATION_VERSION
    if svm_type_enum == _svm.EPSILON_SVR or svm_type_enum == _svm.NU_SVR:
        svm = export_spec.supportVectorRegressor
    else:
        svm = export_spec.supportVectorClassifier
    inferred_length = _infer_min_num_features(libsvm_model)
    if isinstance(feature_names, _string_types):
        if input_length == 'auto':
            print("[WARNING] Infering an input length of %d. If this is not correct, use the 'input_length' parameter." % inferred_length)
            input_length = inferred_length
        elif inferred_length > input_length:
            raise ValueError('An input length of %d was given, but the model requires an input of at least %d.' % (input_length, inferred_length))
        input = export_spec.description.input.add()
        input.name = feature_names
        input.type.multiArrayType.shape.append(input_length)
        input.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.DOUBLE
    else:
        if inferred_length > len(feature_names):
            raise ValueError('%d feature names were given, but the model requires at least %d features.' % (len(feature_names), inferred_length))
        for cur_input_name in feature_names:
            input = export_spec.description.input.add()
            input.name = cur_input_name
            input.type.doubleType.MergeFromString(b'')
    output = export_spec.description.output.add()
    output.name = target
    if svm_type_enum == _svm.EPSILON_SVR or svm_type_enum == _svm.NU_SVR:
        export_spec.description.predictedFeatureName = target
        output.type.doubleType.MergeFromString(b'')
        nr_class = 2
    elif svm_type_enum == _svm.C_SVC or svm_type_enum == _svm.NU_SVC:
        export_spec.description.predictedFeatureName = target
        output.type.int64Type.MergeFromString(b'')
        nr_class = len(libsvm_model.get_labels())
        for i in range(nr_class):
            svm.numberOfSupportVectorsPerClass.append(libsvm_model.nSV[i])
            svm.int64ClassLabels.vector.append(libsvm_model.label[i])
        if probability and bool(libsvm_model.probA):
            output = export_spec.description.output.add()
            output.name = probability
            output.type.dictionaryType.MergeFromString(b'')
            output.type.dictionaryType.int64KeyType.MergeFromString(b'')
            export_spec.description.predictedProbabilitiesName = probability
    else:
        raise ValueError('Only the following SVM types are supported: C_SVC, NU_SVC, EPSILON_SVR, NU_SVR')
    if libsvm_model.param.kernel_type == _svm.LINEAR:
        svm.kernel.linearKernel.MergeFromString(b'')
    elif libsvm_model.param.kernel_type == _svm.RBF:
        svm.kernel.rbfKernel.gamma = libsvm_model.param.gamma
    elif libsvm_model.param.kernel_type == _svm.POLY:
        svm.kernel.polyKernel.degree = libsvm_model.param.degree
        svm.kernel.polyKernel.c = libsvm_model.param.coef0
        svm.kernel.polyKernel.gamma = libsvm_model.param.gamma
    elif libsvm_model.param.kernel_type == _svm.SIGMOID:
        svm.kernel.sigmoidKernel.c = libsvm_model.param.coef0
        svm.kernel.sigmoidKernel.gamma = libsvm_model.param.gamma
    else:
        raise ValueError('Unsupported kernel. The following kernel are supported: linear, RBF, polynomial and sigmoid.')
    if svm_type_enum == _svm.C_SVC or svm_type_enum == _svm.NU_SVC:
        num_class_pairs = nr_class * (nr_class - 1) // 2
        for i in range(num_class_pairs):
            svm.rho.append(libsvm_model.rho[i])
        if bool(libsvm_model.probA) and bool(libsvm_model.probB):
            for i in range(num_class_pairs):
                svm.probA.append(libsvm_model.probA[i])
                svm.probB.append(libsvm_model.probB[i])
    else:
        svm.rho = libsvm_model.rho[0]
    if svm_type_enum == _svm.C_SVC or svm_type_enum == _svm.NU_SVC:
        for _ in range(nr_class - 1):
            svm.coefficients.add()
        for i in range(libsvm_model.l):
            for j in range(nr_class - 1):
                svm.coefficients[j].alpha.append(libsvm_model.sv_coef[j][i])
    else:
        for i in range(libsvm_model.l):
            svm.coefficients.alpha.append(libsvm_model.sv_coef[0][i])
    for i in range(libsvm_model.l):
        j = 0
        cur_support_vector = svm.sparseSupportVectors.vectors.add()
        while libsvm_model.SV[i][j].index != -1:
            cur_node = cur_support_vector.nodes.add()
            cur_node.index = libsvm_model.SV[i][j].index
            cur_node.value = libsvm_model.SV[i][j].value
            j += 1
    model = MLModel(export_spec)
    from libsvm import __version__ as libsvm_version
    libsvm_version = 'libsvm=={0}'.format(libsvm_version)
    model.user_defined_metadata[_METADATA_VERSION] = ct_version
    model.user_defined_metadata[_METADATA_SOURCE] = libsvm_version
    return model