"""Utils for testing the standard gates."""
from inspect import signature, Parameter

def _get_free_params(fun, ignore=None):
    if False:
        i = 10
        return i + 15
    'Get the names of the free parameters of the function ``f``.\n\n    Args:\n        fun (callable): The function to inspect.\n        ignore (list[str]): A list of argument names (as str) to ignore.\n\n    Returns:\n        list[str]: The name of the free parameters not listed in ``ignore``.\n    '
    ignore = ignore or ['kwargs']
    free_params = []
    for (name, param) in signature(fun).parameters.items():
        if param.default == Parameter.empty and param.kind != Parameter.VAR_POSITIONAL:
            if name not in ignore:
                free_params.append(name)
    return free_params