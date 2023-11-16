from typing import Any
from typing import Type
from ...util.logger import debug
from .action_data_empty import ActionDataEmpty
action_types = {}

def action_type_for_type(obj_or_type: Any) -> Type:
    if False:
        for i in range(10):
            print('nop')
    'Convert standard type to Syft types\n\n    Parameters:\n        obj_or_type: Union[object, type]\n            Can be an object or a class\n    '
    if type(obj_or_type) != type:
        if isinstance(obj_or_type, ActionDataEmpty):
            obj_or_type = obj_or_type.syft_internal_type
        else:
            obj_or_type = type(obj_or_type)
    if obj_or_type not in action_types:
        debug(f'WARNING: No Type for {obj_or_type}, returning {action_types[Any]}')
        return action_types[Any]
    return action_types[obj_or_type]

def action_type_for_object(obj: Any) -> Type:
    if False:
        return 10
    'Convert standard type to Syft types\n\n    Parameters:\n        obj_or_type: Union[object, type]\n            Can be an object or a class\n    '
    _type = type(obj)
    if _type not in action_types:
        debug(f'WARNING: No Type for {_type}, returning {action_types[Any]}')
        return action_types[Any]
    return action_types[_type]