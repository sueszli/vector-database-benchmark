MIL_TO_NN_MAPPING_REGISTRY = {}

def register_mil_to_nn_mapping(func=None, override=False):
    if False:
        return 10

    def func_wrapper(_func):
        if False:
            while True:
                i = 10
        f_name = _func.__name__
        if not override and f_name in MIL_TO_NN_MAPPING_REGISTRY:
            raise ValueError('MIL to NN mapping for MIL op {} is already registered.'.format(f_name))
        MIL_TO_NN_MAPPING_REGISTRY[f_name] = _func
        return _func
    if func is None:
        return func_wrapper
    return func_wrapper(func)