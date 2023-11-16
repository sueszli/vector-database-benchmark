_TORCH_OPS_REGISTRY = {}

def register_torch_op(_func=None, torch_alias=None, override=False):
    if False:
        print('Hello World!')
    '\n    Registration routine for PyTorch operators\n    _func: (PyTorch conversion function) [Default=None]\n        PyTorch conversion function to register\n\n    torch_alias: (List of string) [Default=None]\n        All other PyTorch operators that should also be mapped to\n        current conversion routine.\n        e.g. Sort aliased with SortV1, SortV2\n        All provided alias operators must not be registered previously.\n\n    override: (Boolean) [Default=False]\n        If True, overrides earlier registration i.e. specified\n        operator and alias will start pointing to current conversion\n        function.\n        Otherwise, duplicate registration will error out.\n    '

    def func_wrapper(func):
        if False:
            while True:
                i = 10
        f_name = func.__name__
        if not override and f_name in _TORCH_OPS_REGISTRY:
            raise ValueError('Torch Op {} already registered.'.format(f_name))
        _TORCH_OPS_REGISTRY[f_name] = func
        if torch_alias is not None:
            for name in torch_alias:
                if not override and name in _TORCH_OPS_REGISTRY:
                    msg = 'Torch Op alias {} already registered.'
                    raise ValueError(msg.format(name))
                _TORCH_OPS_REGISTRY[name] = func
        return func
    if _func is None:
        return func_wrapper
    return func_wrapper(_func)