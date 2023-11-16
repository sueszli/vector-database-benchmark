from ..._deps import _HAS_LIBSVM

def load_model(model_path):
    if False:
        print('Hello World!')
    'Load a libsvm model from a path on disk.\n\n    This currently supports:\n      * C-SVC\n      * NU-SVC\n      * Epsilon-SVR\n      * NU-SVR\n\n    Parameters\n    ----------\n    model_path: str\n        Path on disk where the libsvm model representation is.\n\n    Returns\n    -------\n    model: libsvm_model\n        A model of the libsvm format.\n    '
    if not _HAS_LIBSVM:
        raise RuntimeError('libsvm not found. libsvm conversion API is disabled.')
    from svmutil import svm_load_model
    import os
    if not os.path.exists(model_path):
        raise IOError('Expected a valid file path. %s does not exist' % model_path)
    return svm_load_model(model_path)