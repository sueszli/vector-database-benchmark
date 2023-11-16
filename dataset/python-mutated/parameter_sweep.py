from itertools import product

def generate_param_grid(params):
    if False:
        return 10
    'Generator of parameter grids.\n    Generate parameter lists from a parameter dictionary in the form of:\n\n    .. code-block:: python\n\n       {\n           "param1": [value1, value2],\n           "param2": [value1, value2]\n       }\n\n    to:\n\n    .. code-block:: python\n\n       [\n           {"param1": value1, "param2": value1},\n           {"param1": value2, "param2": value1},\n           {"param1": value1, "param2": value2},\n           {"param1": value2, "param2": value2}\n       ]\n\n    Args:\n        param_dict (dict): dictionary of parameters and values (in a list).\n\n    Return:\n        list: A list of parameter dictionary string that can be fed directly into\n        model builder as keyword arguments.\n    '
    param_new = {}
    param_fixed = {}
    for (key, value) in params.items():
        if isinstance(value, list):
            param_new[key] = value
        else:
            param_fixed[key] = value
    items = sorted(param_new.items())
    (keys, values) = zip(*items)
    params_exp = []
    for v in product(*values):
        param_exp = dict(zip(keys, v))
        param_exp.update(param_fixed)
        params_exp.append(param_exp)
    return params_exp