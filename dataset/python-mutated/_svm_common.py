"""
Common stuff for SVMs
"""

def _set_kernel(model, spec):
    if False:
        while True:
            i = 10
    '\n    Takes the sklearn SVM model and returns the spec with the protobuf kernel for that model.\n    '

    def gamma_value(model):
        if False:
            return 10
        if model.gamma == 'auto' or model.gamma == 'auto_deprecated':
            return 1 / float(len(model.support_vectors_[0]))
        else:
            return model.gamma
    result = None
    if model.kernel == 'linear':
        spec.kernel.linearKernel.MergeFromString(b'')
    elif model.kernel == 'rbf':
        spec.kernel.rbfKernel.gamma = gamma_value(model)
    elif model.kernel == 'poly':
        spec.kernel.polyKernel.gamma = gamma_value(model)
        spec.kernel.polyKernel.c = model.coef0
        spec.kernel.polyKernel.degree = model.degree
    elif model.kernel == 'sigmoid':
        spec.kernel.sigmoidKernel.gamma = gamma_value(model)
        spec.kernel.sigmoidKernel.c = model.coef0
    else:
        raise ValueError('Unsupported kernel. The following kernel are supported: linear, RBF, polynomial and sigmoid.')
    return result