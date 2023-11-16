from typing import Mapping

def deep_update(from_dict, to_dict):
    if False:
        print('Hello World!')
    for (key, value) in from_dict.items():
        if key in to_dict.keys() and isinstance(to_dict[key], Mapping) and isinstance(value, Mapping):
            deep_update(value, to_dict[key])
        else:
            to_dict[key] = value