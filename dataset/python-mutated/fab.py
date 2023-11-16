from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.security.permissions import ACTION_CAN_ACCESS_MENU, ACTION_CAN_CREATE, ACTION_CAN_DELETE, ACTION_CAN_EDIT, ACTION_CAN_READ
if TYPE_CHECKING:
    from airflow.auth.managers.base_auth_manager import ResourceMethod
_MAP_METHOD_NAME_TO_FAB_ACTION_NAME: dict[ResourceMethod, str] = {'POST': ACTION_CAN_CREATE, 'GET': ACTION_CAN_READ, 'PUT': ACTION_CAN_EDIT, 'DELETE': ACTION_CAN_DELETE}

def get_fab_action_from_method_map():
    if False:
        while True:
            i = 10
    'Returns the map associating a method to a FAB action.'
    return _MAP_METHOD_NAME_TO_FAB_ACTION_NAME

def get_method_from_fab_action_map():
    if False:
        print('Hello World!')
    'Returns the map associating a FAB action to a method.'
    return {**{v: k for (k, v) in _MAP_METHOD_NAME_TO_FAB_ACTION_NAME.items()}, ACTION_CAN_ACCESS_MENU: 'GET'}