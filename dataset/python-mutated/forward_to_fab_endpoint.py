from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
from airflow.api_connexion.exceptions import BadRequest
from airflow.auth.managers.fab.api_endpoints import role_and_permission_endpoint, user_endpoint
from airflow.www.extensions.init_auth_manager import get_auth_manager
if TYPE_CHECKING:
    from typing import Callable
    from airflow.api_connexion.types import APIResponse

def _require_fab(func: Callable) -> Callable:
    if False:
        print('Hello World!')
    '\n    Raise an HTTP error 400 if the auth manager is not FAB.\n\n    Intended to decorate endpoints that have been migrated from Airflow API to FAB API.\n    '

    def inner(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        from airflow.auth.managers.fab.fab_auth_manager import FabAuthManager
        auth_mgr = get_auth_manager()
        if not isinstance(auth_mgr, FabAuthManager):
            raise BadRequest(detail='This endpoint is only available when using the default auth manager FabAuthManager.')
        else:
            warnings.warn('This API endpoint is deprecated. Please use the API under /auth/fab/v1 instead for this operation.', DeprecationWarning)
            return func(*args, **kwargs)
    return inner

@_require_fab
def get_role(**kwargs) -> APIResponse:
    if False:
        i = 10
        return i + 15
    'Get role.'
    return role_and_permission_endpoint.get_role(**kwargs)

@_require_fab
def get_roles(**kwargs) -> APIResponse:
    if False:
        while True:
            i = 10
    'Get roles.'
    return role_and_permission_endpoint.get_roles(**kwargs)

@_require_fab
def delete_role(**kwargs) -> APIResponse:
    if False:
        return 10
    'Delete a role.'
    return role_and_permission_endpoint.delete_role(**kwargs)

@_require_fab
def patch_role(**kwargs) -> APIResponse:
    if False:
        for i in range(10):
            print('nop')
    'Update a role.'
    kwargs.pop('body', None)
    return role_and_permission_endpoint.patch_role(**kwargs)

@_require_fab
def post_role(**kwargs) -> APIResponse:
    if False:
        for i in range(10):
            print('nop')
    'Create a new role.'
    kwargs.pop('body', None)
    return role_and_permission_endpoint.post_role(**kwargs)

@_require_fab
def get_permissions(**kwargs) -> APIResponse:
    if False:
        for i in range(10):
            print('nop')
    'Get permissions.'
    return role_and_permission_endpoint.get_permissions(**kwargs)

@_require_fab
def get_user(**kwargs) -> APIResponse:
    if False:
        for i in range(10):
            print('nop')
    'Get a user.'
    return user_endpoint.get_user(**kwargs)

@_require_fab
def get_users(**kwargs) -> APIResponse:
    if False:
        for i in range(10):
            print('nop')
    'Get users.'
    return user_endpoint.get_users(**kwargs)

@_require_fab
def post_user(**kwargs) -> APIResponse:
    if False:
        return 10
    'Create a new user.'
    kwargs.pop('body', None)
    return user_endpoint.post_user(**kwargs)

@_require_fab
def patch_user(**kwargs) -> APIResponse:
    if False:
        for i in range(10):
            print('nop')
    'Update a user.'
    kwargs.pop('body', None)
    return user_endpoint.patch_user(**kwargs)

@_require_fab
def delete_user(**kwargs) -> APIResponse:
    if False:
        for i in range(10):
            print('nop')
    'Delete a user.'
    return user_endpoint.delete_user(**kwargs)