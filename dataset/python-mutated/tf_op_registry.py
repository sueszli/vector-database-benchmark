_TF_OPS_REGISTRY = {}

def register_tf_op(_func=None, tf_alias=None, override=False):
    if False:
        while True:
            i = 10
    '\n    Registration routine for TensorFlow operators\n    _func: (TF conversion function) [Default=None]\n        TF conversion function to register\n\n    tf_alias: (List of string) [Default=None]\n        All other TF operators that should also be mapped to\n        current conversion routine.\n        e.g. Sort aliased with SortV1, SortV2\n        All provided alias operators must not be registered previously.\n\n    override: (Boolean) [Default=False]\n        If True, overrides earlier registration i.e. specified\n        operator and alias will start pointing to current conversion\n        function.\n        Otherwise, duplicate registration will error out.\n    '

    def func_wrapper(func):
        if False:
            i = 10
            return i + 15
        f_name = func.__name__
        if not override and f_name in _TF_OPS_REGISTRY:
            raise ValueError('TF op {} already registered.'.format(f_name))
        _TF_OPS_REGISTRY[f_name] = func
        if tf_alias is not None:
            for name in tf_alias:
                if not override and name in _TF_OPS_REGISTRY:
                    msg = 'TF op alias {} already registered.'
                    raise ValueError(msg.format(name))
                _TF_OPS_REGISTRY[name] = func
        return func
    if _func is None:
        return func_wrapper
    return func_wrapper(_func)