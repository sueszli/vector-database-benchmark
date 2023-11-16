"""Test roles"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import json
import os
import warnings

import pytest
from pytest import mark
from sqlalchemy.exc import SADeprecationWarning
from tornado.log import app_log

from .. import orm, roles
from ..scopes import get_scopes_for, scope_definitions
from ..utils import utcnow
from .mocking import MockHub
from .utils import add_user, api_request


@mark.role
def test_orm_roles(db):
    """Test orm roles setup"""
    user_role = orm.Role.find(db, name='user')
    token_role = orm.Role.find(db, name='token')
    service_role = orm.Role.find(db, name='service')
    if not user_role:
        user_role = orm.Role(name='user', scopes=['self'])
        db.add(user_role)
    if not token_role:
        token_role = orm.Role(name='token', scopes=['inherit'])
        db.add(token_role)
    if not service_role:
        service_role = orm.Role(name='service', scopes=[])
        db.add(service_role)
    db.commit()

    group_role = orm.Role(name='group', scopes=['read:users'])
    db.add(group_role)
    db.commit()

    user = orm.User(name='falafel')
    db.add(user)
    db.commit()

    service = orm.Service(name='kebab')
    db.add(service)
    db.commit()

    group = orm.Group(name='fast-food')
    db.add(group)
    db.commit()

    assert user_role.users == []
    assert user_role.services == []
    assert user_role.groups == []
    assert service_role.users == []
    assert service_role.services == []
    assert service_role.groups == []
    assert user.roles == []
    assert service.roles == []
    assert group.roles == []

    user_role.users.append(user)
    service_role.services.append(service)
    group_role.groups.append(group)
    db.commit()
    assert user_role.users == [user]
    assert user.roles == [user_role]
    assert service_role.services == [service]
    assert service.roles == [service_role]
    assert group_role.groups == [group]
    assert group.roles == [group_role]

    # check token creation without specifying its role
    # assigns it the default 'token' role
    token = user.new_api_token()
    user_token = orm.APIToken.find(db, token=token)
    assert set(user_token.scopes) == set(token_role.scopes)

    # check creating token with a specific role
    token = service.new_api_token(roles=['service'])
    service_token = orm.APIToken.find(db, token=token)
    assert set(service_token.scopes) == set(service_role.scopes)

    # check deleting user removes the user from roles
    db.delete(user)
    db.commit()
    assert user_role.users == []
    # check deleting the service_role removes it from service.roles
    db.delete(service_role)
    db.commit()
    assert service.roles == []
    # check deleting the group removes it from group_roles
    db.delete(group)
    db.commit()
    assert group_role.groups == []

    # clean up db
    db.delete(service)
    db.delete(group_role)
    db.commit()


@mark.role
def test_orm_roles_delete_cascade(db):
    """Orm roles cascade"""
    user1 = orm.User(name='user1')
    user2 = orm.User(name='user2')
    role1 = orm.Role(name='role1')
    role2 = orm.Role(name='role2')
    db.add(user1)
    db.add(user2)
    db.add(role1)
    db.add(role2)
    db.commit()
    # add user to role via user.roles
    user1.roles.append(role1)
    db.commit()
    assert user1 in role1.users
    assert role1 in user1.roles

    # add user to role via roles.users
    role1.users.append(user2)
    db.commit()
    assert user2 in role1.users
    assert role1 in user2.roles

    # fill role2 and check role1 again
    role2.users.append(user1)
    role2.users.append(user2)
    db.commit()
    assert user1 in role1.users
    assert user2 in role1.users
    assert user1 in role2.users
    assert user2 in role2.users
    assert role1 in user1.roles
    assert role1 in user2.roles
    assert role2 in user1.roles
    assert role2 in user2.roles

    # now start deleting
    # 1. remove role via user.roles
    user1.roles.remove(role2)
    db.commit()
    assert user1 not in role2.users
    assert role2 not in user1.roles

    # 2. remove user via role.users
    role1.users.remove(user2)
    db.commit()
    assert user2 not in role1.users
    assert role1 not in user2.roles

    # 3. delete role object
    db.delete(role2)
    db.commit()
    assert role2 not in user1.roles
    assert role2 not in user2.roles

    # 4. delete user object
    db.delete(user1)
    db.delete(user2)
    db.commit()
    assert user1 not in role1.users


@mark.role
@mark.parametrize(
    "scopes, expected_scopes",
    [
        (
            ['admin:users'],
            {
                'admin:users',
                'admin:auth_state',
                'users',
                'delete:users',
                'list:users',
                'read:users',
                'users:activity',
                'read:users:name',
                'read:users:groups',
                'read:roles:users',
                'read:users:activity',
            },
        ),
        (
            ['users'],
            {
                'users',
                'list:users',
                'read:users',
                'users:activity',
                'read:users:name',
                'read:users:groups',
                'read:users:activity',
            },
        ),
        (
            ['read:users'],
            {
                'read:users',
                'read:users:name',
                'read:users:groups',
                'read:users:activity',
            },
        ),
        (['read:servers'], {'read:servers', 'read:users:name'}),
        (
            ['admin:groups'],
            {
                'admin:groups',
                'groups',
                'delete:groups',
                'list:groups',
                'read:groups',
                'read:roles:groups',
                'read:groups:name',
            },
        ),
        (
            ['admin:groups', 'read:servers'],
            {
                'admin:groups',
                'groups',
                'delete:groups',
                'list:groups',
                'read:groups',
                'read:roles:groups',
                'read:groups:name',
                'read:servers',
                'read:users:name',
            },
        ),
        (
            ['tokens!group=hobbits'],
            {'tokens!group=hobbits', 'read:tokens!group=hobbits'},
        ),
        (
            ['admin:services'],
            {
                'read:roles:services',
                'read:services:name',
                'admin:services',
                'list:services',
                'read:services',
            },
        ),
    ],
)
def test_get_expanded_scopes(db, scopes, expected_scopes):
    """Test role scopes expansion into their fully expanded scopes"""
    roles.create_role(db, {'name': 'testing_scopes', 'scopes': scopes})
    role = orm.Role.find(db, name='testing_scopes')
    expanded_scopes = roles.roles_to_expanded_scopes([role], owner=None)
    assert expanded_scopes == expected_scopes
    db.delete(role)


@mark.role
async def test_load_default_roles(tmpdir, request):
    """Test loading default roles in app.py"""
    kwargs = {}
    ssl_enabled = getattr(request.module, "ssl_enabled", False)
    if ssl_enabled:
        kwargs['internal_certs_location'] = str(tmpdir)
    hub = MockHub(**kwargs)
    hub.init_db()
    db = hub.db
    await hub.init_role_creation()
    # test default roles loaded to database
    default_roles = roles.get_default_roles()
    for role in default_roles:
        assert orm.Role.find(db, role['name']) is not None


@mark.role
@mark.parametrize(
    "role, role_def, response_type, response",
    [
        (
            'new-role',
            {
                'name': 'new-role',
                'description': 'Some description',
                'scopes': ['groups'],
            },
            'info',
            app_log.info('Role new-role added to database'),
        ),
        (
            'the-same-role',
            {
                'name': 'new-role',
                'description': 'Some description',
                'scopes': ['groups'],
            },
            'no-log',
            None,
        ),
        ('no_name', {'scopes': ['users']}, 'error', KeyError),
        (
            'no_scopes',
            {'name': 'no-permissions'},
            'warning',
            app_log.warning('Warning: New defined role no-permissions has no scopes'),
        ),
        (
            'admin',
            {'name': 'admin', 'scopes': ['admin:users']},
            'error',
            ValueError,
        ),
        (
            'admin',
            {'name': 'admin', 'description': 'New description'},
            'error',
            ValueError,
        ),
        (
            'user',
            {'name': 'user', 'scopes': ['read:users:name']},
            'info',
            app_log.info('Role user scopes attribute has been changed'),
        ),
        # rewrite the user role back to 'default'
        (
            'user',
            {'name': 'user', 'scopes': ['self']},
            'info',
            app_log.info('Role user scopes attribute has been changed'),
        ),
    ],
)
async def test_creating_roles(app, role, role_def, response_type, response):
    """Test raising errors and warnings when creating/modifying roles"""

    db = app.db

    if response_type == 'error':
        with pytest.raises(response):
            roles.create_role(db, role_def)

    elif response_type == 'warning' or response_type == 'info':
        with pytest.warns(response):
            roles.create_role(db, role_def)
        # check the role has been created/modified
        role = orm.Role.find(db, role_def['name'])
        assert role is not None
        if 'description' in role_def.keys():
            assert role.description == role_def['description']
        if 'scopes' in role_def.keys():
            assert role.scopes == role_def['scopes']

    # make sure no warnings/info logged when the role exists and its definition hasn't been changed
    elif response_type == 'no-log':
        with pytest.warns(response) as record:
            # don't catch already-suppressed sqlalchemy warnings
            warnings.simplefilter("ignore", SADeprecationWarning)
            roles.create_role(db, role_def)

        for warning in record.list:
            # show warnings for debugging
            print("Unexpected warning", warning)
        assert not record.list
        role = orm.Role.find(db, role_def['name'])
        assert role is not None


@mark.role
@mark.parametrize(
    "role_type, rolename, response_type, response",
    [
        (
            'existing',
            'test-role1',
            'info',
            app_log.info('Role user scopes attribute has been changed'),
        ),
        ('non-existing', 'test-role2', 'error', KeyError),
        ('default', 'user', 'error', ValueError),
    ],
)
async def test_delete_roles(db, role_type, rolename, response_type, response):
    """Test raising errors and info when deleting roles"""

    if response_type == 'info':
        # add the role to db
        test_role = orm.Role(name=rolename)
        db.add(test_role)
        db.commit()
        check_role = orm.Role.find(db, rolename)
        assert check_role is not None
        # check the role is deleted and info raised
        with pytest.warns(response):
            roles.delete_role(db, rolename)
        check_role = orm.Role.find(db, rolename)
        assert check_role is None

    elif response_type == 'error':
        with pytest.raises(response):
            roles.delete_role(db, rolename)


@mark.role
@mark.parametrize(
    "role, response",
    [
        (
            {
                'name': 'test-scopes-1',
                'scopes': [
                    'users',
                    'users!user=charlie',
                    'admin:groups',
                    'read:tokens',
                ],
            },
            'existing',
        ),
        ({'name': 'test-scopes-2', 'scopes': ['uses']}, KeyError),
        ({'name': 'test-scopes-3', 'scopes': ['users:activities']}, KeyError),
        ({'name': 'test-scopes-4', 'scopes': ['groups!goup=class-A']}, KeyError),
    ],
)
async def test_scope_existence(tmpdir, request, role, response):
    """Test checking of scopes provided in role definitions"""
    kwargs = {'load_roles': [role]}
    ssl_enabled = getattr(request.module, "ssl_enabled", False)
    if ssl_enabled:
        kwargs['internal_certs_location'] = str(tmpdir)
    hub = MockHub(**kwargs)
    hub.init_db()
    db = hub.db

    if response == 'existing':
        roles.create_role(db, role)
        added_role = orm.Role.find(db, role['name'])
        assert added_role is not None
        assert added_role.scopes == role['scopes']

    elif response == KeyError:
        with pytest.raises(response):
            roles.create_role(db, role)
        added_role = orm.Role.find(db, role['name'])
        assert added_role is None

    # delete the tested roles
    if added_role:
        roles.delete_role(db, added_role.name)


@mark.role
@mark.parametrize(
    "explicit_allowed_users",
    [
        (True,),
        (False,),
    ],
)
async def test_load_roles_users(tmpdir, request, explicit_allowed_users):
    """Test loading predefined roles for users in app.py"""
    roles_to_load = [
        {
            'name': 'teacher',
            'description': 'Access users information, servers and groups without create/delete privileges',
            'scopes': ['users', 'groups'],
            'users': ['cyclops', 'gandalf'],
        },
    ]
    kwargs = {'load_roles': roles_to_load}
    ssl_enabled = getattr(request.module, "ssl_enabled", False)
    if ssl_enabled:
        kwargs['internal_certs_location'] = str(tmpdir)
    hub = MockHub(**kwargs)
    hub.init_db()
    db = hub.db
    hub.authenticator.admin_users = ['admin']
    if explicit_allowed_users:
        hub.authenticator.allowed_users = ['cyclops', 'gandalf', 'bilbo', 'gargamel']
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_role_assignment()
    admin_role = orm.Role.find(db, 'admin')
    user_role = orm.Role.find(db, 'user')
    # test if every user has a role (and no duplicates)
    # and admins have admin role
    for user in db.query(orm.User):
        assert len(user.roles) > 0
        assert len(user.roles) == len(set(user.roles))
        if user.admin:
            assert admin_role in user.roles
            assert user_role in user.roles

    # test if predefined roles loaded and assigned
    teacher_role = orm.Role.find(db, name='teacher')
    assert teacher_role is not None
    gandalf_user = orm.User.find(db, name='gandalf')
    assert teacher_role in gandalf_user.roles
    cyclops_user = orm.User.find(db, name='cyclops')
    assert teacher_role in cyclops_user.roles

    # delete the test roles
    for role in roles_to_load:
        roles.delete_role(db, role['name'])


@mark.role
async def test_load_roles_services(tmpdir, request, preserve_scopes):
    services = [
        {'name': 'idle-culler', 'api_token': 'some-token'},
        {'name': 'user_service', 'api_token': 'some-other-token'},
        {'name': 'admin_service', 'api_token': 'secret-token'},
    ]
    service_tokens = {
        'some-token': 'idle-culler',
        'some-other-token': 'user_service',
        'secret-token': 'admin_service',
    }
    custom_scopes = {
        "custom:empty-scope": {
            "description": "empty custom scope",
        }
    }
    roles_to_load = [
        {
            'name': 'idle-culler',
            'description': 'Cull idle servers',
            'scopes': [
                'read:users:name',
                'read:users:activity',
                'read:servers',
                'servers',
                'custom:empty-scope',
            ],
            'services': ['idle-culler'],
        },
    ]
    kwargs = {
        'custom_scopes': custom_scopes,
        'load_roles': roles_to_load,
        'services': services,
        'service_tokens': service_tokens,
    }
    ssl_enabled = getattr(request.module, "ssl_enabled", False)
    if ssl_enabled:
        kwargs['internal_certs_location'] = str(tmpdir)
    hub = MockHub(**kwargs)
    hub.init_db()
    db = hub.db
    await hub.init_role_creation()
    await hub.init_api_tokens()
    # make 'admin_service' admin
    admin_service = orm.Service.find(db, 'admin_service')
    admin_service.admin = True
    db.commit()
    await hub.init_role_assignment()
    # test if every service has a role (and no duplicates)
    admin_role = orm.Role.find(db, name='admin')
    user_role = orm.Role.find(db, name='user')

    # test if predefined roles loaded and assigned
    culler_role = orm.Role.find(db, name='idle-culler')
    culler_service = orm.Service.find(db, name='idle-culler')
    assert culler_service.roles == [culler_role]
    user_service = orm.Service.find(db, name='user_service')
    assert not user_service.roles
    assert admin_service.roles == [admin_role]

    # delete the test services
    for service in db.query(orm.Service):
        db.delete(service)
    db.commit()

    # delete the test tokens
    for token in db.query(orm.APIToken):
        db.delete(token)
    db.commit()

    # delete the test roles
    for role in roles_to_load:
        roles.delete_role(db, role['name'])


@mark.role
async def test_load_roles_groups(tmpdir, request):
    """Test loading predefined roles for groups in app.py"""
    groups_to_load = {
        'group1': {'users': ['gandalf'], 'properties': {}},
        'group2': {'users': ['bilbo', 'gargamel'], 'properties': {}},
        'group3': {'users': ['cyclops'], 'properties': {}},
    }
    roles_to_load = [
        {
            'name': 'assistant',
            'description': 'Access users information only',
            'scopes': ['read:users'],
            'groups': ['group2'],
        },
        {
            'name': 'head',
            'description': 'Whole user access',
            'scopes': ['users', 'admin:users'],
            'groups': ['group3', "group4"],
        },
    ]
    kwargs = {'load_groups': groups_to_load, 'load_roles': roles_to_load}
    ssl_enabled = getattr(request.module, "ssl_enabled", False)
    if ssl_enabled:
        kwargs['internal_certs_location'] = str(tmpdir)
    hub = MockHub(**kwargs)
    hub.init_db()
    db = hub.db
    await hub.init_role_creation()
    await hub.init_groups()
    await hub.init_role_assignment()

    assist_role = orm.Role.find(db, name='assistant')
    head_role = orm.Role.find(db, name='head')

    group1 = orm.Group.find(db, name='group1')
    group2 = orm.Group.find(db, name='group2')
    group3 = orm.Group.find(db, name='group3')
    group4 = orm.Group.find(db, name='group4')

    # test group roles
    assert group1.roles == []
    assert group2 in assist_role.groups
    assert group3 in head_role.groups
    assert group4 in head_role.groups

    # delete the test roles
    for role in roles_to_load:
        roles.delete_role(db, role['name'])


@mark.role
async def test_load_roles_user_tokens(tmpdir, request):
    user_tokens = {
        'secret-token': 'cyclops',
        'secrety-token': 'gandalf',
        'super-secret-token': 'admin',
    }
    roles_to_load = [
        {
            'name': 'reader',
            'description': 'Read all users models',
            'scopes': ['read:users'],
        },
    ]
    kwargs = {
        'load_roles': roles_to_load,
        'api_tokens': user_tokens,
    }
    ssl_enabled = getattr(request.module, "ssl_enabled", False)
    if ssl_enabled:
        kwargs['internal_certs_location'] = str(tmpdir)
    hub = MockHub(**kwargs)
    hub.init_db()
    db = hub.db
    hub.authenticator.admin_users = ['admin']
    hub.authenticator.allowed_users = ['cyclops', 'gandalf']
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_api_tokens()
    await hub.init_role_assignment()
    # test if all other tokens have default 'user' role
    token_role = orm.Role.find(db, 'token')
    secret_token = orm.APIToken.find(db, 'secret-token')
    assert set(secret_token.scopes) == set(token_role.scopes)
    secrety_token = orm.APIToken.find(db, 'secrety-token')
    assert set(secrety_token.scopes) == set(token_role.scopes)

    # delete the test tokens
    for token in db.query(orm.APIToken):
        db.delete(token)
    db.commit()

    # delete the test roles
    for role in roles_to_load:
        roles.delete_role(db, role['name'])


@mark.role
@mark.parametrize(
    "headers, rolename, scopes, status",
    [
        # no role requested - gets default 'token' role
        ({}, None, None, 201),
        # role scopes within the user's default 'user' role
        ({}, 'self-reader', ['read:users!user'], 201),
        # role scopes within the user's default 'user' role, but with disjoint filter
        ({}, 'other-reader', ['read:users!user=other'], 400),
        # role scopes within the user's default 'user' role, without filter
        ({}, 'other-reader', ['read:users'], 400),
        # role scopes outside of the user's role but within the group's role scopes of which the user is a member
        ({}, 'groups-reader', ['read:groups'], 201),
        # non-existing role request
        ({}, 'non-existing', [], 400),
        # role scopes outside of both user's role and group's role scopes
        ({}, 'users-creator', ['admin:users'], 400),
    ],
)
async def test_get_new_token_via_api(app, headers, rolename, scopes, status):
    """Test requesting a token via API with and without roles"""

    user = add_user(app.db, app, name='user')
    if rolename and rolename != 'non-existing':
        roles.create_role(app.db, {'name': rolename, 'scopes': scopes})
        if rolename == 'groups-reader':
            # add role for a group
            roles.create_role(app.db, {'name': 'group-role', 'scopes': ['groups']})
            # create a group and add the user and group_role
            group = orm.Group.find(app.db, 'test-group')
            if not group:
                group = orm.Group(name='test-group')
                app.db.add(group)
                group_role = orm.Role.find(app.db, 'group-role')
                group.roles.append(group_role)
                user.groups.append(group)
                app.db.commit()
    if rolename:
        body = json.dumps({'roles': [rolename]})
    else:
        body = ''
    # request a new token
    r = await api_request(
        app, 'users/user/tokens', method='post', headers=headers, data=body
    )
    assert r.status_code == status
    if status != 200:
        return
    # check the new-token reply for roles
    reply = r.json()
    assert 'token' in reply
    assert reply['user'] == user.name
    if not rolename:
        assert reply['roles'] == ['token']
    else:
        assert reply['roles'] == [rolename]
    token_id = reply['id']

    # delete the token
    r = await api_request(app, 'users/user/tokens', token_id, method='delete')
    assert r.status_code == 204
    # verify deletion
    r = await api_request(app, 'users/user/tokens', token_id)
    assert r.status_code == 404


@mark.role
@mark.parametrize(
    "kind, has_user_scopes",
    [
        ('users', True),
        ('services', False),
    ],
)
async def test_self_expansion(app, kind, has_user_scopes):
    Class = orm.get_class(kind)
    orm_obj = Class(name=f'test_{kind}')
    app.db.add(orm_obj)
    app.db.commit()
    test_role = orm.Role(name='test_role', scopes=['self'])
    orm_obj.roles.append(test_role)
    # test expansion of user/service scopes
    scopes = get_scopes_for(orm_obj)
    assert bool(scopes) == has_user_scopes
    if kind == 'users':
        for scope in scopes:
            assert scope.endswith(f"!user={orm_obj.name}")
            base_scope = scope.split("!", 1)[0]
            assert base_scope in scope_definitions

    # test expansion of token scopes
    orm_obj.new_api_token()
    print(orm_obj.api_tokens[0])
    token_scopes = get_scopes_for(orm_obj.api_tokens[0])
    print(token_scopes)
    assert bool(token_scopes) == has_user_scopes
    app.db.delete(orm_obj)
    app.db.delete(test_role)


@mark.role
@mark.parametrize(
    "scope_list, kind, test_for_token",
    [
        (['users:activity!user'], 'users', False),
        (['users:activity!user', 'read:users'], 'users', False),
        (['users:activity!user=otheruser', 'read:users'], 'users', False),
        (['users:activity!user'], 'users', True),
        (['users:activity!user=otheruser', 'groups'], 'users', True),
    ],
)
async def test_user_filter_expansion(app, scope_list, kind, test_for_token):
    Class = orm.get_class(kind)
    orm_obj = Class(name=f'test_{kind}')
    app.db.add(orm_obj)
    app.db.commit()

    test_role = orm.Role(name='test_role', scopes=scope_list)
    orm_obj.roles.append(test_role)

    if test_for_token:
        token = orm_obj.new_api_token(roles=['test_role'])
        orm_token = orm.APIToken.find(app.db, token)
        expanded_scopes = get_scopes_for(orm_token)
    else:
        expanded_scopes = get_scopes_for(orm_obj)

    for scope in scope_list:
        base, _, filter = scope.partition('!')
        for ex_scope in expanded_scopes:
            ex_base, ex__, ex_filter = ex_scope.partition('!')
            # check that the filter has been expanded to include the username if '!user' filter
            if scope in ex_scope and filter == 'user':
                assert ex_filter == f'{filter}={orm_obj.name}'
            # make sure the filter has been left unchanged if other filter provided
            elif scope in ex_scope and '=' in filter:
                assert ex_filter == filter

    app.db.delete(orm_obj)
    app.db.delete(test_role)


@mark.role
@mark.parametrize(
    "name, valid",
    [
        ('abc', True),
        ('group', True),
        ("a-pretty-long-name-with-123", True),
        ("0-abc", False),  # starts with number
        ("role-", False),  # ends with -
        ("has-Uppercase", False),  # Uppercase
        ("a" * 256, False),  # too long
        ("has space", False),  # space is illegal
    ],
)
async def test_valid_names(name, valid):
    if valid:
        assert roles._validate_role_name(name)
    else:
        with pytest.raises(ValueError):
            roles._validate_role_name(name)


@mark.role
async def test_server_token_role(app):
    user = add_user(app.db, app, name='test_user')
    assert user.api_tokens == []
    spawner = user.spawner
    spawner.cmd = ['jupyterhub-singleuser']
    await user.spawn()

    server_token = spawner.api_token
    orm_server_token = orm.APIToken.find(app.db, server_token)
    assert orm_server_token

    # resolve `!server` filter in server role
    server_role_scopes = {
        s.replace("!server", f"!server={user.name}/")
        for s in orm.Role.find(app.db, "server").scopes
    }
    assert set(orm_server_token.scopes) == server_role_scopes

    assert orm_server_token.user.name == user.name
    assert user.api_tokens == [orm_server_token]

    await user.stop()


@mark.role
@mark.parametrize(
    "token_role, api_method, api_endpoint, for_user, response",
    [
        ('server', 'post', 'activity', 'same_user', 200),
        ('server', 'post', 'activity', 'other_user', 404),
        ('server', 'get', 'users', 'same_user', 403),
        ('token', 'post', 'activity', 'same_user', 200),
        ('no_role', 'post', 'activity', 'same_user', 403),
    ],
)
async def test_server_role_api_calls(
    app, token_role, api_method, api_endpoint, for_user, response
):
    user = add_user(app.db, app, name='test_user')
    roles.grant_role(app.db, user, 'user')
    app_log.debug(user.roles)
    app_log.debug(get_scopes_for(user.orm_user))
    if token_role == 'no_role':
        api_token = user.new_api_token(roles=[])
    else:
        api_token = user.new_api_token(roles=[token_role])

    if for_user == 'same_user':
        username = user.name
    else:
        username = 'otheruser'

    if api_endpoint == 'activity':
        path = f"users/{username}/activity"
        data = json.dumps({"servers": {"": {"last_activity": utcnow().isoformat()}}})
    elif api_endpoint == 'users':
        path = "users"
        data = ""

    r = await api_request(
        app,
        path,
        headers={"Authorization": f"token {api_token}"},
        data=data,
        method=api_method,
    )
    assert r.status_code == response


async def test_oauth_client_allowed_scopes(app):
    allowed_scopes = ['read:users', 'read:groups']
    service = {
        'name': 'oas1',
        'api_token': 'some-token',
        'oauth_client_allowed_scopes': allowed_scopes,
    }
    app.services.append(service)
    app.init_services()
    app_service = app.services[0]
    assert app_service['name'] == 'oas1'
    assert set(app_service['oauth_client_allowed_scopes']) == set(allowed_scopes)


async def test_user_group_roles(app, create_temp_role):
    user = add_user(app.db, app, name='jack')
    another_user = add_user(app.db, app, name='jill')

    group = orm.Group.find(app.db, name='A')
    if not group:
        group = orm.Group(name='A')
        app.db.add(group)
        app.db.commit()

    if group not in user.groups:
        user.groups.append(group)
        app.db.commit()

    if group not in another_user.groups:
        another_user.groups.append(group)
        app.db.commit()

    group_role = orm.Role.find(app.db, 'student-a')
    if not group_role:
        create_temp_role(['read:groups!group=A', 'list:groups!group=A'], 'student-a')
        roles.grant_role(app.db, group, rolename='student-a')
        group_role = orm.Role.find(app.db, 'student-a')

    # repeat check to ensure group roles don't get added to the user at all
    # regression test for #3472
    roles_before = list(user.roles)
    for i in range(3):
        get_scopes_for(user.orm_user)
        user_roles = list(user.roles)
        assert user_roles == roles_before

    # jack's API token
    token = user.new_api_token()

    headers = {'Authorization': 'token %s' % token}
    r = await api_request(app, f'users/{user.name}', method='get', headers=headers)
    assert r.status_code == 200
    r.raise_for_status()
    reply = r.json()

    print(reply)

    assert reply['name'] == 'jack'
    assert len(reply['roles']) == 1
    assert group_role.name not in reply['roles']

    headers = {'Authorization': 'token %s' % token}
    r = await api_request(app, 'groups', method='get', headers=headers)
    assert r.status_code == 200
    r.raise_for_status()
    reply = r.json()

    print(reply)
    assert len(reply) == 1
    assert reply[0]['name'] == 'A'

    headers = {'Authorization': 'token %s' % token}
    r = await api_request(app, f'users/{user.name}', method='get', headers=headers)
    assert r.status_code == 200
    r.raise_for_status()
    reply = r.json()

    print(reply)

    assert reply['name'] == 'jack'
    assert len(reply['roles']) == 1
    assert group_role.name not in reply['roles']


async def test_config_role_list():
    roles_to_load = [
        {
            'name': 'elephant',
            'description': 'pacing about',
            'scopes': ['read:hub'],
        },
        {
            'name': 'tiger',
            'description': 'pouncing stuff',
            'scopes': ['shutdown'],
        },
    ]
    hub = MockHub(load_roles=roles_to_load)
    hub.init_db()
    hub.authenticator.admin_users = ['admin']
    await hub.init_role_creation()
    for role_conf in roles_to_load:
        assert orm.Role.find(hub.db, name=role_conf['name'])
    # Now remove elephant from config and see if it is removed from database
    roles_to_load.pop(0)
    hub.load_roles = roles_to_load
    await hub.init_role_creation()
    assert orm.Role.find(hub.db, name='tiger')
    assert not orm.Role.find(hub.db, name='elephant')


async def test_config_role_users():
    role_name = 'painter'
    user_name = 'benny'
    user_names = ['agnetha', 'bjorn', 'anni-frid', user_name]
    roles_to_load = [
        {
            'name': role_name,
            'description': 'painting with colors',
            'scopes': ['users', 'groups'],
            'users': user_names,
        },
    ]
    hub = MockHub(load_roles=roles_to_load)
    hub.init_db()
    hub.authenticator.admin_users = ['admin']
    hub.authenticator.allowed_users = user_names
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_role_assignment()
    user = orm.User.find(hub.db, name=user_name)
    role = orm.Role.find(hub.db, name=role_name)
    assert role in user.roles
    # Now reload and see if user is removed from role list
    roles_to_load[0]['users'].remove(user_name)
    hub.load_roles = roles_to_load
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_role_assignment()
    user = orm.User.find(hub.db, name=user_name)
    role = orm.Role.find(hub.db, name=role_name)
    assert role not in user.roles


async def test_duplicate_role_users():
    role_name = 'painter'
    user_name = 'benny'
    user_names = ['agnetha', 'bjorn', 'anni-frid', user_name]
    roles_to_load = [
        {
            'name': role_name,
            'description': 'painting with colors',
            'scopes': ['users', 'groups'],
            'users': user_names,
        },
        {
            'name': role_name,
            'description': 'painting with colors',
            'scopes': ['users', 'groups'],
            'users': user_names,
        },
    ]
    hub = MockHub(load_roles=roles_to_load)
    hub.init_db()
    with pytest.raises(ValueError):
        await hub.init_role_creation()


async def test_admin_role_and_flag():
    admin_role_spec = [
        {
            'name': 'admin',
            'users': ['eddy'],
        }
    ]
    hub = MockHub(load_roles=admin_role_spec)
    hub.init_db()
    hub.authenticator.admin_users = ['admin']
    hub.authenticator.allowed_users = ['eddy']
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_role_assignment()
    admin_role = orm.Role.find(hub.db, name='admin')
    for user_name in ['eddy', 'admin']:
        user = orm.User.find(hub.db, name=user_name)
        assert user.admin
        assert admin_role in user.roles
    admin_role_spec[0]['users'].remove('eddy')
    hub.load_roles = admin_role_spec
    await hub.init_users()
    await hub.init_role_assignment()
    user = orm.User.find(hub.db, name='eddy')
    assert not user.admin
    assert admin_role not in user.roles


async def test_custom_role_reset():
    user_role_spec = [
        {
            'name': 'user',
            'scopes': ['self', 'shutdown'],
            'users': ['eddy'],
        }
    ]
    hub = MockHub(load_roles=user_role_spec)
    hub.init_db()
    hub.authenticator.allowed_users = ['eddy']
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_role_assignment()
    user_role = orm.Role.find(hub.db, name='user')
    user = orm.User.find(hub.db, name='eddy')
    assert user_role in user.roles
    assert 'shutdown' in user_role.scopes
    hub.load_roles = []
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_role_assignment()
    user_role = orm.Role.find(hub.db, name='user')
    user = orm.User.find(hub.db, name='eddy')
    assert user_role in user.roles
    assert 'shutdown' not in user_role.scopes


async def test_removal_config_to_db():
    role_spec = [
        {
            'name': 'user',
            'scopes': ['self', 'shutdown'],
        },
        {
            'name': 'wizard',
            'scopes': ['self', 'read:groups'],
        },
    ]
    hub = MockHub(load_roles=role_spec)
    hub.init_db()
    await hub.init_role_creation()
    assert orm.Role.find(hub.db, 'user')
    assert orm.Role.find(hub.db, 'wizard')
    hub.load_roles = []
    await hub.init_role_creation()
    assert orm.Role.find(hub.db, 'user')
    assert not orm.Role.find(hub.db, 'wizard')


async def test_no_admin_role_change():
    role_spec = [{'name': 'admin', 'scopes': ['shutdown']}]
    hub = MockHub(load_roles=role_spec)
    hub.init_db()
    with pytest.raises(ValueError):
        await hub.init_role_creation()


@pytest.mark.parametrize(
    "in_db, role_users, allowed_users, expected_members",
    [
        # users in the db, not specified in custom user role
        # no change to membership
        (["alpha", "beta"], None, None, ["alpha", "beta"]),
        # allowed_users is additive, not strict
        (["alpha", "beta"], None, {"gamma"}, ["alpha", "beta", "gamma"]),
        # explicit empty revokes all assignments
        (["alpha", "beta"], [], None, []),
        # explicit value is respected exactly
        (["alpha", "beta"], ["alpha", "gamma"], None, ["alpha", "gamma"]),
    ],
)
async def test_user_role_from_config(
    in_db, role_users, allowed_users, expected_members
):
    role_spec = {
        'name': 'user',
        'scopes': ['self', 'shutdown'],
    }
    if role_users is not None:
        role_spec['users'] = role_users
    hub = MockHub(load_roles=[role_spec])
    hub.init_db()
    db = hub.db
    hub.authenticator.admin_users = set()
    if allowed_users:
        hub.authenticator.allowed_users = allowed_users
    await hub.init_role_creation()


async def test_user_config_creates_default_role():
    role_spec = [
        {
            'name': 'new-role',
            'scopes': ['read:users'],
            'users': ['not-yet-created-user'],
        }
    ]
    user_names = []
    hub = MockHub(load_roles=role_spec)
    hub.init_db()
    hub.authenticator.allowed_users = user_names
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_role_assignment()
    user_role = orm.Role.find(hub.db, 'user')
    new_role = orm.Role.find(hub.db, 'new-role')
    assert orm.User.find(hub.db, 'not-yet-created-user') in new_role.users
    assert orm.User.find(hub.db, 'not-yet-created-user') in user_role.users


async def test_admin_role_respects_config():
    role_spec = [
        {
            'name': 'admin',
        }
    ]
    admin_users = ['eddy', 'carol']
    hub = MockHub(load_roles=role_spec)
    hub.init_db()
    hub.authenticator.admin_users = admin_users
    await hub.init_role_creation()
    await hub.init_users()
    await hub.init_role_assignment()
    admin_role = orm.Role.find(hub.db, 'admin')
    for user_name in admin_users:
        user = orm.User.find(hub.db, user_name)
        assert user in admin_role.users


@pytest.mark.parametrize(
    "in_db, role_users, admin_users, expected_members",
    [
        # users in the db, not specified in custom user role
        # no change to membership
        (["alpha", "beta"], None, None, ["alpha", "beta"]),
        # admin_users is additive, not strict
        (["alpha", "beta"], None, {"gamma"}, ["alpha", "beta", "gamma"]),
        # explicit empty revokes all assignments
        (["alpha", "beta"], [], None, []),
        # explicit value is respected exactly
        (["alpha", "beta"], ["alpha", "gamma"], None, ["alpha", "gamma"]),
    ],
)
async def test_admin_role_membership(in_db, role_users, admin_users, expected_members):
    load_roles = []
    if role_users is not None:
        load_roles.append({"name": "admin", "users": role_users})
    if not admin_users:
        admin_users = set()
    hub = MockHub(load_roles=load_roles, db_url="sqlite:///:memory:")
    hub.init_db()
    await hub.init_role_creation()
    db = hub.db
    hub.authenticator.admin_users = admin_users
    # add in_db users to the database
    # this is the 'before' state of who had the role before startup
    for username in in_db or []:
        user = orm.User(name=username)
        db.add(user)
        db.commit()
        roles.grant_role(db, user, "admin")
    db.commit()
    await hub.init_users()
    await hub.init_role_assignment()
    admin_role = orm.Role.find(db, 'admin')
    role_members = sorted(user.name for user in admin_role.users)
    assert role_members == expected_members


async def test_no_default_service_role():
    services = [
        {
            'name': 'minesweeper',
            'api_token': 'some-token',
        }
    ]
    hub = MockHub(services=services)
    await hub.initialize()
    service = orm.Service.find(hub.db, 'minesweeper')
    assert not service.roles


async def test_hub_upgrade_detection(tmpdir):
    db_url = f"sqlite:///{tmpdir.join('jupyterhub.sqlite')}"
    os.environ['JUPYTERHUB_TEST_DB_URL'] = db_url
    # Create hub with users and tokens
    hub = MockHub(db_url=db_url)
    await hub.initialize()
    user_names = ['patricia', 'quentin']
    user_role = orm.Role.find(hub.db, 'user')
    for name in user_names:
        user = add_user(hub.db, name=name)
        user.new_api_token()
        assert user_role in user.roles
    for role in hub.db.query(orm.Role):
        hub.db.delete(role)
    hub.db.commit()
    # Restart hub in emulated upgrade mode: default roles for all entities
    hub.test_clean_db = False
    await hub.initialize()
    assert getattr(hub, '_rbac_upgrade', False)
    user_role = orm.Role.find(hub.db, 'user')
    token_role = orm.Role.find(hub.db, 'token')
    for name in user_names:
        user = orm.User.find(hub.db, name)
        assert user_role in user.roles
        assert set(user.api_tokens[0].scopes) == set(token_role.scopes)
    # Strip all roles and see if it sticks
    user_role.users = []
    token_role.tokens = []
    hub.db.commit()

    hub.init_db()
    hub.init_hub()
    await hub.init_role_creation()
    await hub.init_users()
    hub.authenticator.allowed_users = ['patricia']
    await hub.init_api_tokens()
    await hub.init_role_assignment()
    assert not getattr(hub, '_rbac_upgrade', False)
    user_role = orm.Role.find(hub.db, 'user')
    token_role = orm.Role.find(hub.db, 'token')
    allowed_user = orm.User.find(hub.db, 'patricia')
    rem_user = orm.User.find(hub.db, 'quentin')
    assert user_role in allowed_user.roles
    assert user_role not in rem_user.roles
    assert token_role not in rem_user.roles


async def test_login_default_role(app, username):
    cookies = await app.login_user(username)
    user = app.users[username]
    # assert login new user gets 'user' role
    assert [role.name for role in user.roles] == ["user"]

    # clear roles, keep user
    user.roles = []
    app.db.commit()

    # login *again*; user exists,
    # login should always trigger "user" role assignment
    cookies = await app.login_user(username)
    user = app.users[username]
    assert [role.name for role in user.roles] == ["user"]
