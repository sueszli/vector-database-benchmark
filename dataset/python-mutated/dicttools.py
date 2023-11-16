"""Tools for working with dicts."""
from typing import Any, Dict, Mapping, Optional

def _unflatten_single_dict(flat_dict):
    if False:
        return 10
    "Convert a flat dict of key-value pairs to dict tree.\n\n    Example\n    -------\n\n        _unflatten_single_dict({\n          foo_bar_baz: 123,\n          foo_bar_biz: 456,\n          x_bonks: 'hi',\n        })\n\n        # Returns:\n        # {\n        #   foo: {\n        #     bar: {\n        #       baz: 123,\n        #       biz: 456,\n        #     },\n        #   },\n        #   x: {\n        #     bonks: 'hi'\n        #   }\n        # }\n\n    Parameters\n    ----------\n    flat_dict : dict\n        A one-level dict where keys are fully-qualified paths separated by\n        underscores.\n\n    Returns\n    -------\n    dict\n        A tree made of dicts inside of dicts.\n\n    "
    out: Dict[str, Any] = dict()
    for (pathstr, v) in flat_dict.items():
        path = pathstr.split('_')
        prev_dict: Optional[Dict[str, Any]] = None
        curr_dict = out
        for k in path:
            if k not in curr_dict:
                curr_dict[k] = dict()
            prev_dict = curr_dict
            curr_dict = curr_dict[k]
        if prev_dict is not None:
            prev_dict[k] = v
    return out

def unflatten(flat_dict, encodings=None):
    if False:
        print('Hello World!')
    "Converts a flat dict of key-value pairs to a spec tree.\n\n    Example\n    -------\n        unflatten({\n          foo_bar_baz: 123,\n          foo_bar_biz: 456,\n          x_bonks: 'hi',\n        }, ['x'])\n\n        # Returns:\n        # {\n        #   foo: {\n        #     bar: {\n        #       baz: 123,\n        #       biz: 456,\n        #     },\n        #   },\n        #   encoding: {  # This gets added automatically\n        #     x: {\n        #       bonks: 'hi'\n        #     }\n        #   }\n        # }\n\n    Args\n    ----\n    flat_dict: dict\n        A flat dict where keys are fully-qualified paths separated by\n        underscores.\n\n    encodings: set\n        Key names that should be automatically moved into the 'encoding' key.\n\n    Returns\n    -------\n    A tree made of dicts inside of dicts.\n    "
    if encodings is None:
        encodings = set()
    out_dict = _unflatten_single_dict(flat_dict)
    for (k, v) in list(out_dict.items()):
        if isinstance(v, dict):
            v = unflatten(v, encodings)
        elif hasattr(v, '__iter__'):
            for (i, child) in enumerate(v):
                if isinstance(child, dict):
                    v[i] = unflatten(child, encodings)
        if k in encodings:
            if 'encoding' not in out_dict:
                out_dict['encoding'] = dict()
            out_dict['encoding'][k] = v
            out_dict.pop(k)
    return out_dict

def remove_none_values(input_dict: Mapping[Any, Any]) -> Dict[Any, Any]:
    if False:
        for i in range(10):
            print('nop')
    'Remove all keys with None values from a dict.'
    new_dict = {}
    for (key, val) in input_dict.items():
        if isinstance(val, dict):
            val = remove_none_values(val)
        if val is not None:
            new_dict[key] = val
    return new_dict