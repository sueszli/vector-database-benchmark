from __future__ import annotations
import functools
import inspect
import importlib
from collections import defaultdict, OrderedDict
from logging import getLogger
from typing import Any, Callable, Collection, KeysView, Optional, Union
from types import ModuleType
from ckan.common import config, current_user
import ckan.plugins as p
import ckan.model as model
from ckan.common import _
from ckan.types import AuthResult, AuthFunction, DataDict, Context
log = getLogger(__name__)

def get_local_functions(module: ModuleType, include_private: bool=False):
    if False:
        for i in range(10):
            print('nop')
    'Return list of (name, func) tuples.\n\n    Filters out all non-callables and all the items that were\n    imported.\n    '
    return inspect.getmembers(module, lambda func: inspect.isfunction(func) and inspect.getmodule(func) is module and (include_private or not func.__name__.startswith('_')))

class AuthFunctions:
    """ This is a private cache used by get_auth_function() and should never be
    accessed directly we will create an instance of it and then remove it."""
    _functions: dict[str, AuthFunction] = {}

    def clear(self) -> None:
        if False:
            print('Hello World!')
        ' clear any stored auth functions. '
        self._functions.clear()

    def keys(self) -> KeysView[str]:
        if False:
            for i in range(10):
                print('nop')
        ' Return a list of known auth functions.'
        if not self._functions:
            self._build()
        return self._functions.keys()

    def get(self, function: str) -> Optional[AuthFunction]:
        if False:
            while True:
                i = 10
        ' Return the requested auth function. '
        if not self._functions:
            self._build()
        return self._functions.get(function)

    @staticmethod
    def _is_chained_auth_function(func: AuthFunction) -> bool:
        if False:
            print('Hello World!')
        '\n        Helper function to check if a function is a chained auth function, i.e.\n        it has been decorated with the chain auth function decorator.\n        '
        return getattr(func, 'chained_auth_function', False)

    def _build(self) -> None:
        if False:
            i = 10
            return i + 15
        'Gather the auth functions.\n\n        First get the default ones in the ckan/logic/auth directory\n        Rather than writing them out in full will use\n        importlib.import_module to load anything from ckan.auth that\n        looks like it might be an authorisation function\n\n        '
        module_root = 'ckan.logic.auth'
        for auth_module_name in ['get', 'create', 'update', 'delete', 'patch']:
            module = importlib.import_module('.' + auth_module_name, module_root)
            for (key, v) in get_local_functions(module):
                if not hasattr(v, 'auth_allow_anonymous_access'):
                    if auth_module_name == 'get':
                        v.auth_allow_anonymous_access = True
                    else:
                        v.auth_allow_anonymous_access = False
                self._functions[key] = v
        resolved_auth_function_plugins: dict[str, str] = {}
        fetched_auth_functions = {}
        chained_auth_functions = defaultdict(list)
        for plugin in p.PluginImplementations(p.IAuthFunctions):
            for (name, auth_function) in plugin.get_auth_functions().items():
                if self._is_chained_auth_function(auth_function):
                    chained_auth_functions[name].append(auth_function)
                elif name in resolved_auth_function_plugins:
                    raise Exception('The auth function %r is already implemented in %r' % (name, resolved_auth_function_plugins[name]))
                else:
                    resolved_auth_function_plugins[name] = plugin.name
                    fetched_auth_functions[name] = auth_function
        for (name, func_list) in chained_auth_functions.items():
            if name not in fetched_auth_functions and name not in self._functions:
                raise Exception('The auth %r is not found for chained auth' % name)
            for func in reversed(func_list):
                if name in fetched_auth_functions:
                    prev_func = fetched_auth_functions[name]
                else:
                    prev_func = self._functions[name]
                new_func = functools.partial(func, prev_func)
                for (attribute, value) in func.__dict__.items():
                    setattr(new_func, attribute, value)
                fetched_auth_functions[name] = new_func
        self._functions.update(fetched_auth_functions)
_AuthFunctions = AuthFunctions()
del AuthFunctions

def clear_auth_functions_cache() -> None:
    if False:
        while True:
            i = 10
    _AuthFunctions.clear()

def auth_functions_list() -> KeysView[str]:
    if False:
        print('Hello World!')
    'Returns a list of the names of the auth functions available.  Currently\n    this is to allow the Auth Audit to know if an auth function is available\n    for a given action.'
    return _AuthFunctions.keys()

def is_sysadmin(username: Optional[str]) -> bool:
    if False:
        while True:
            i = 10
    ' Returns True is username is a sysadmin '
    user = _get_user(username)
    return bool(user and user.sysadmin)

def _get_user(username: Optional[str]) -> Optional['model.User']:
    if False:
        while True:
            i = 10
    '\n    Try to get the user from current_user proxy, if possible.\n    If not fallback to using the DB\n    '
    if not username:
        return None
    try:
        if current_user.name == username:
            return current_user
    except AttributeError:
        pass
    except TypeError:
        pass
    except RuntimeError:
        pass
    return model.User.get(username)

def get_group_or_org_admin_ids(group_id: Optional[str]) -> list[str]:
    if False:
        while True:
            i = 10
    if not group_id:
        return []
    group = model.Group.get(group_id)
    if not group:
        return []
    q = model.Session.query(model.Member.table_id).filter(model.Member.group_id == group.id).filter(model.Member.table_name == 'user').filter(model.Member.state == 'active').filter(model.Member.capacity == 'admin')
    return [a.table_id for a in q]

def is_authorized_boolean(action: str, context: Context, data_dict: Optional[DataDict]=None) -> bool:
    if False:
        print('Hello World!')
    ' runs the auth function but just returns True if allowed else False\n    '
    outcome = is_authorized(action, context, data_dict=data_dict)
    return outcome.get('success', False)

def is_authorized(action: str, context: Context, data_dict: Optional[DataDict]=None) -> AuthResult:
    if False:
        print('Hello World!')
    if context.get('ignore_auth'):
        return {'success': True}
    auth_function = _AuthFunctions.get(action)
    if auth_function:
        username = context.get('user')
        user = _get_user(username)
        if user:
            if user.is_deleted():
                return {'success': False}
            elif user.sysadmin:
                if not getattr(auth_function, 'auth_sysadmins_check', False):
                    return {'success': True}
        if not getattr(auth_function, 'auth_allow_anonymous_access', False) and (not context.get('auth_user_obj')):
            if isinstance(auth_function, functools.partial):
                name = auth_function.func.__name__
            else:
                name = auth_function.__name__
            return {'success': False, 'msg': 'Action {0} requires an authenticated user'.format(name)}
        return auth_function(context, data_dict or {})
    else:
        raise ValueError(_('Authorization function not found: %s' % action))
ROLE_PERMISSIONS: dict[str, list[str]] = OrderedDict([('admin', ['admin', 'membership']), ('editor', ['read', 'delete_dataset', 'create_dataset', 'update_dataset', 'manage_group']), ('member', ['read', 'manage_group'])])

def get_collaborator_capacities() -> Collection[str]:
    if False:
        i = 10
        return i + 15
    if check_config_permission('allow_admin_collaborators'):
        return ('admin', 'editor', 'member')
    else:
        return ('editor', 'member')
_trans_functions: dict[str, Callable[[], str]] = {'admin': lambda : _('Admin'), 'editor': lambda : _('Editor'), 'member': lambda : _('Member')}

def trans_role(role: str) -> str:
    if False:
        i = 10
        return i + 15
    return _trans_functions[role]()

def roles_list() -> list[dict[str, str]]:
    if False:
        print('Hello World!')
    ' returns list of roles for forms '
    roles = []
    for role in ROLE_PERMISSIONS:
        roles.append(dict(text=trans_role(role), value=role))
    return roles

def roles_trans() -> dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    ' return dict of roles with translation '
    roles = {}
    for role in ROLE_PERMISSIONS:
        roles[role] = trans_role(role)
    return roles

def get_roles_with_permission(permission: str) -> list[str]:
    if False:
        print('Hello World!')
    ' returns the roles with the permission requested '
    roles = []
    for role in ROLE_PERMISSIONS:
        permissions = ROLE_PERMISSIONS[role]
        if permission in permissions or 'admin' in permissions:
            roles.append(role)
    return roles

def has_user_permission_for_group_or_org(group_id: Optional[str], user_name: Optional[str], permission: str) -> bool:
    if False:
        return 10
    ' Check if the user has the given permissions for the group, allowing for\n    sysadmin rights and permission cascading down a group hierarchy.\n\n    '
    if not group_id:
        return False
    group = model.Group.get(group_id)
    if not group:
        return False
    group_id = group.id
    if is_sysadmin(user_name):
        return True
    user_id = get_user_id_for_username(user_name, allow_none=True)
    if not user_id:
        return False
    if _has_user_permission_for_groups(user_id, permission, [group_id]):
        return True
    capacities = check_config_permission('roles_that_cascade_to_sub_groups')
    assert isinstance(capacities, list)
    for capacity in capacities:
        parent_groups = group.get_parent_group_hierarchy(type=group.type)
        group_ids = [group_.id for group_ in parent_groups]
        if _has_user_permission_for_groups(user_id, permission, group_ids, capacity=capacity):
            return True
    return False

def _has_user_permission_for_groups(user_id: str, permission: str, group_ids: list[str], capacity: Optional[str]=None) -> bool:
    if False:
        for i in range(10):
            print('nop')
    ' Check if the user has the given permissions for the particular\n    group (ignoring permissions cascading in a group hierarchy).\n    Can also be filtered by a particular capacity.\n    '
    if not group_ids:
        return False
    q: Any = model.Session.query(model.Member.capacity).filter(model.Member.group_id.in_(group_ids)).filter(model.Member.table_name == 'user').filter(model.Member.state == 'active').filter(model.Member.table_id == user_id)
    if capacity:
        q = q.filter(model.Member.capacity == capacity)
    for row in q:
        perms = ROLE_PERMISSIONS.get(row.capacity, [])
        if 'admin' in perms or permission in perms:
            return True
    return False

def users_role_for_group_or_org(group_id: Optional[str], user_name: Optional[str]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    " Returns the user's role for the group. (Ignores privileges that cascade\n    in a group hierarchy.)\n\n    "
    if not group_id:
        return None
    group = model.Group.get(group_id)
    if not group:
        return None
    user_id = get_user_id_for_username(user_name, allow_none=True)
    if not user_id:
        return None
    q: Any = model.Session.query(model.Member.capacity).filter(model.Member.group_id == group.id).filter(model.Member.table_name == 'user').filter(model.Member.state == 'active').filter(model.Member.table_id == user_id)
    for row in q:
        return row.capacity
    return None

def has_user_permission_for_some_org(user_name: Optional[str], permission: str) -> bool:
    if False:
        i = 10
        return i + 15
    ' Check if the user has the given permission for any organization. '
    user_id = get_user_id_for_username(user_name, allow_none=True)
    if not user_id:
        return False
    roles = get_roles_with_permission(permission)
    if not roles:
        return False
    q: Any = model.Session.query(model.Member.group_id).filter(model.Member.table_name == 'user').filter(model.Member.state == 'active').filter(model.Member.capacity.in_(roles)).filter(model.Member.table_id == user_id)
    group_ids = []
    for row in q:
        group_ids.append(row.group_id)
    if not group_ids:
        return False
    permission_exists: bool = model.Session.query(model.Session.query(model.Group).filter(model.Group.is_organization == True).filter(model.Group.state == 'active').filter(model.Group.id.in_(group_ids)).exists()).scalar()
    return permission_exists

def get_user_id_for_username(user_name: Optional[str], allow_none: bool=False) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    ' Helper function to get user id '
    user = _get_user(user_name)
    if user:
        return user.id
    if allow_none:
        return None
    raise Exception('Not logged in user')

def can_manage_collaborators(package_id: str, user_id: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if a user is allowed to manage the collaborators of a given\n    dataset.\n\n    Currently a user can manage collaborators if:\n\n    1. Is an administrator of the organization the dataset belongs to\n    2. Is a collaborator with role "admin" (\n        assuming :ref:`ckan.auth.allow_admin_collaborators` is set to True)\n    3. Is the creator of the dataset and the dataset does not belong to an\n        organization (\n        requires :ref:`ckan.auth.create_dataset_if_not_in_organization`\n        and :ref:`ckan.auth.create_unowned_dataset`)\n    '
    pkg = model.Package.get(package_id)
    if not pkg:
        return False
    owner_org = pkg.owner_org
    if not owner_org and check_config_permission('create_dataset_if_not_in_organization') and check_config_permission('create_unowned_dataset') and (pkg.creator_user_id == user_id):
        return True
    if has_user_permission_for_group_or_org(owner_org, user_id, 'membership'):
        return True
    return user_is_collaborator_on_dataset(user_id, pkg.id, 'admin')

def user_is_collaborator_on_dataset(user_id: str, dataset_id: str, capacity: Optional[Union[str, list[str]]]=None) -> bool:
    if False:
        while True:
            i = 10
    '\n    Returns True if the provided user is a collaborator on the provided\n    dataset.\n\n    If capacity is provided it restricts the check to the capacity\n    provided (eg `admin` or `editor`). Multiple capacities can be\n    provided passing a list\n\n    '
    q = model.Session.query(model.PackageMember).filter(model.PackageMember.user_id == user_id).filter(model.PackageMember.package_id == dataset_id)
    if capacity:
        if isinstance(capacity, str):
            capacity = [capacity]
        q = q.filter(model.PackageMember.capacity.in_(capacity))
    return model.Session.query(q.exists()).scalar()
CONFIG_PERMISSIONS_DEFAULTS: dict[str, Union[bool, str]] = {'anon_create_dataset': False, 'create_dataset_if_not_in_organization': True, 'create_unowned_dataset': True, 'user_create_groups': True, 'user_create_organizations': True, 'user_delete_groups': True, 'user_delete_organizations': True, 'create_user_via_api': False, 'create_user_via_web': True, 'roles_that_cascade_to_sub_groups': 'admin', 'public_activity_stream_detail': False, 'allow_dataset_collaborators': False, 'allow_admin_collaborators': False, 'allow_collaborators_to_change_owner_org': False, 'create_default_api_keys': False}

def check_config_permission(permission: str) -> Union[list[str], bool]:
    if False:
        while True:
            i = 10
    'Returns the configuration value for the provided permission\n\n    Permission is a string indentifying the auth permission (eg\n    `anon_create_dataset`), optionally prefixed with `ckan.auth.`.\n\n    The possible values for `permission` are the keys of\n    CONFIG_PERMISSIONS_DEFAULTS. These can be overriden in the config file\n    by prefixing them with `ckan.auth.`.\n\n    Returns the permission value, generally True or False, except on\n    `roles_that_cascade_to_sub_groups` which is a list of strings.\n\n    '
    key = permission.replace('ckan.auth.', '')
    if key not in CONFIG_PERMISSIONS_DEFAULTS:
        return False
    config_key = 'ckan.auth.' + key
    value = config.get(config_key)
    return value

def auth_is_anon_user(context: Context) -> bool:
    if False:
        for i in range(10):
            print('nop')
    ' Is this an anonymous user?\n        eg Not logged in if a web request and not user defined in context\n        if logic functions called directly\n\n        See ckan/lib/base.py:232 for pylons context object logic\n    '
    context_user = context.get('user')
    is_anon_user = not bool(context_user)
    return is_anon_user