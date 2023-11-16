from typing import Dict, Tuple
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from ._traverse import traverse_state_dict, set_element, OBJ_PATH, STATE_DICT_ITEM
'\nTODO:\nNeed to add ability to handle tuple, OrderedDict, NamedTuple.\nUpdate mappings from dict to a class.\nChange set_element to recreate the right type for tuple, OrderedDict, and NamedTuple.\n'
FLATTEN_MAPPING = Dict[str, OBJ_PATH]

def flatten_state_dict(state_dict: STATE_DICT_TYPE) -> Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.\n\n    Use ``unflatten_state_dict`` to revert this process.\n    Returns:\n        A tuple with the flatten state_dict and a mapping from original to new state_dict.\n    N.B. The new keys are derived from the object paths, joined by dot.\n        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.\n    "
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if False:
            i = 10
            return i + 15
        new_fqn = '.'.join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f'duplicated flatten key {new_fqn}')
        flattened[new_fqn] = value
        mappings[new_fqn] = path
    traverse_state_dict(state_dict, flat_copy)
    return (flattened, mappings)

def unflatten_state_dict(state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING) -> STATE_DICT_TYPE:
    if False:
        print('Hello World!')
    'Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``.'
    nested: STATE_DICT_TYPE = {}
    for (key, value) in state_dict.items():
        set_element(nested, mapping[key], value)
    return nested