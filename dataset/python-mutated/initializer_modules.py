import torch
from ludwig.constants import TYPE
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import initializer_registry

def _create_and_init(init_fn, init_kwargs, *args, **kwargs):
    if False:
        print('Hello World!')
    t = torch.empty(*args, **kwargs)
    init_fn(t, **init_kwargs)
    return t

def get_initializer(parameters):
    if False:
        print('Hello World!')
    if parameters is None:
        return lambda *args, **kwargs: _create_and_init(initializer_registry[parameters], {}, *args, **kwargs)
    elif isinstance(parameters, str):
        initializer_fun = get_from_registry(parameters, initializer_registry)
        return lambda *args, **kwargs: _create_and_init(initializer_fun, {}, *args, **kwargs)
    elif isinstance(parameters, dict):
        initializer_fun = get_from_registry(parameters[TYPE], initializer_registry)
        init_kwargs = parameters.copy()
        del init_kwargs[TYPE]
        return lambda *args, **kwargs: _create_and_init(initializer_fun, init_kwargs, *args, **kwargs)
    else:
        raise ValueError(f'Initializers parameters should be either strings or dictionaries, but the provided parameters are a {type(parameters)}. Parameters values: {parameters}')