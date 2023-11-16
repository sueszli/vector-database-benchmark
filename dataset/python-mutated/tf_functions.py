"""Functions to deal with tf.function objects."""
import dataclasses
from typing import Any, Dict, Tuple, Union, Iterator, List, Optional
DataClassType = Any
LeafType = Any
NestedDict = Dict[str, Union[Any, DataClassType, 'NestedDict']]
KeyList = List[str]

def iterate_nested(nd: NestedDict, previous_keys: Optional[KeyList]=None) -> Iterator[Tuple[KeyList, LeafType]]:
    if False:
        print('Hello World!')
    "Creates an iterator over every leaf value in depth first order.\n\n  Iterates over a nested dictionary in depth first order. The order in which\n  the peer keys are traversed is not guaranteed (same as for the keys of a\n  dictionary).\n\n  ```Example\n  nested_dict = {'a': 1, 'b': [2, 3, 4], 'c': {'d': 8}}\n  for k, v in iterate_nested(nested_dict):\n    print('_'.join(k), v)\n  # Prints out:\n  # a: 1\n  # b: [2, 3, 4]\n  # c_d: 8\n  ```\n\n  Args:\n    nd: The dictionary to be traversed.\n    previous_keys: If supplied, the computed key list will be a join of the\n      previous_keys and the current keys.\n\n  Yields:\n    A tuple of the key path and the value for each leaf node.\n  "
    if previous_keys is None:
        previous_keys = []
    for (k, v) in nd.items():
        keys = previous_keys + [k]
        if not _is_nested(v):
            yield (keys, v)
        else:
            as_dict = dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
            for val in iterate_nested(as_dict, keys):
                yield val

def _is_nested(x: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Returns whether a value is nested.'
    return isinstance(x, dict) or dataclasses.is_dataclass(x)