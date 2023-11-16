import collections.abc
from typing import Any
from immutabledict import immutabledict

def freeze(o: Any) -> Any:
    if False:
        return 10
    if isinstance(o, dict):
        return immutabledict({k: freeze(v) for (k, v) in o.items()})
    if isinstance(o, immutabledict):
        return o
    if isinstance(o, (bytes, str)):
        return o
    try:
        return tuple((freeze(i) for i in o))
    except TypeError:
        pass
    return o

def unfreeze(o: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(o, collections.abc.Mapping):
        return {k: unfreeze(v) for (k, v) in o.items()}
    if isinstance(o, (bytes, str)):
        return o
    try:
        return [unfreeze(i) for i in o]
    except TypeError:
        pass
    return o