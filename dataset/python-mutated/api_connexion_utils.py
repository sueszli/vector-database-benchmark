from __future__ import annotations
from contextlib import contextmanager
from airflow.api_connexion.exceptions import EXCEPTIONS_LINK_MAP
from airflow.auth.managers.fab.security_manager.constants import EXISTING_ROLES

@contextmanager
def create_test_client(app, user_name, role_name, permissions):
    if False:
        while True:
            i = 10
    '\n    Helper function to create a client with a temporary user which will be deleted once done\n    '
    client = app.test_client()
    with create_user_scope(app, username=user_name, role_name=role_name, permissions=permissions) as _:
        resp = client.post('/login/', data={'username': user_name, 'password': user_name})
        assert resp.status_code == 302
        yield client

@contextmanager
def create_user_scope(app, username, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Helper function designed to be used with pytest fixture mainly.\n    It will create a user and provide it for the fixture via YIELD (generator)\n    then will tidy up once test is complete\n    '
    test_user = create_user(app, username, **kwargs)
    try:
        yield test_user
    finally:
        delete_user(app, username)

def create_user(app, username, role_name=None, email=None, permissions=None):
    if False:
        return 10
    appbuilder = app.appbuilder
    delete_user(app, username)
    role = None
    if role_name:
        delete_role(app, role_name)
        role = create_role(app, role_name, permissions)
    else:
        role = []
    return appbuilder.sm.add_user(username=username, first_name=username, last_name=username, email=email or f'{username}@example.org', role=role, password=username)

def create_role(app, name, permissions=None):
    if False:
        return 10
    appbuilder = app.appbuilder
    role = appbuilder.sm.find_role(name)
    if not role:
        role = appbuilder.sm.add_role(name)
    if not permissions:
        permissions = []
    for permission in permissions:
        perm_object = appbuilder.sm.get_permission(*permission)
        appbuilder.sm.add_permission_to_role(role, perm_object)
    return role

def set_user_single_role(app, user, role_name):
    if False:
        print('Hello World!')
    role = create_role(app, role_name)
    if role not in user.roles:
        user.roles = [role]
        app.appbuilder.sm.update_user(user)
        user._perms = None

def delete_role(app, name):
    if False:
        i = 10
        return i + 15
    if name not in EXISTING_ROLES:
        if app.appbuilder.sm.find_role(name):
            app.appbuilder.sm.delete_role(name)

def delete_roles(app):
    if False:
        print('Hello World!')
    for role in app.appbuilder.sm.get_all_roles():
        delete_role(app, role.name)

def delete_user(app, username):
    if False:
        i = 10
        return i + 15
    appbuilder = app.appbuilder
    for user in appbuilder.sm.get_all_users():
        if user.username == username:
            _ = [delete_role(app, role.name) for role in user.roles if role and role.name not in EXISTING_ROLES]
            appbuilder.sm.del_register_user(user)
            break

def delete_users(app):
    if False:
        print('Hello World!')
    for user in app.appbuilder.sm.get_all_users():
        delete_user(app, user.username)

def assert_401(response):
    if False:
        for i in range(10):
            print('nop')
    assert response.status_code == 401, f'Current code: {response.status_code}'
    assert response.json == {'detail': None, 'status': 401, 'title': 'Unauthorized', 'type': EXCEPTIONS_LINK_MAP[401]}