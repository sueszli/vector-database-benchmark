"""Tests for service authentication"""
import copy
import os
import sys
from binascii import hexlify
from unittest import mock
from urllib.parse import parse_qs, urlparse
import pytest
from bs4 import BeautifulSoup
from pytest import raises
from tornado.httputil import url_concat
from tornado.log import app_log
from .. import orm, roles, scopes
from ..roles import roles_to_scopes
from ..services.auth import _ExpiringDict
from ..utils import url_path_join
from .mocking import public_url
from .utils import AsyncSession, async_requests
monotonic_future = mock.patch('time.monotonic', lambda : sys.maxsize)
ssl_enabled = False

def test_expiring_dict():
    if False:
        print('Hello World!')
    cache = _ExpiringDict(max_age=30)
    cache['key'] = 'cached value'
    assert 'key' in cache
    assert cache['key'] == 'cached value'
    with raises(KeyError):
        cache['nokey']
    with monotonic_future:
        assert 'key' not in cache
    cache['key'] = 'cached value'
    assert 'key' in cache
    with monotonic_future:
        assert 'key' not in cache
    cache['key'] = 'cached value'
    assert 'key' in cache
    with monotonic_future:
        with raises(KeyError):
            cache['key']
    cache['key'] = 'cached value'
    assert 'key' in cache
    with monotonic_future:
        assert cache.get('key', 'default') == 'default'
    cache.max_age = 0
    cache['key'] = 'cached value'
    assert 'key' in cache
    with monotonic_future:
        assert cache.get('key', 'default') == 'cached value'

async def test_hubauth_token(app, mockservice_url, create_user_with_scopes):
    """Test HubAuthenticated service with user API tokens"""
    u = create_user_with_scopes('access:services')
    token = u.new_api_token()
    no_access_token = u.new_api_token(roles=[])
    app.db.commit()
    r = await async_requests.get(public_url(app, mockservice_url) + '/whoami/', headers={'Authorization': f'token {no_access_token}'})
    assert r.status_code == 403
    r = await async_requests.get(public_url(app, mockservice_url) + '/whoami/', headers={'Authorization': f'token {token}'})
    r.raise_for_status()
    reply = r.json()
    sub_reply = {key: reply.get(key, 'missing') for key in ['name', 'admin']}
    assert sub_reply == {'name': u.name, 'admin': False}
    r = await async_requests.get(public_url(app, mockservice_url) + '/whoami/?token=%s' % token)
    r.raise_for_status()
    reply = r.json()
    sub_reply = {key: reply.get(key, 'missing') for key in ['name', 'admin']}
    assert sub_reply == {'name': u.name, 'admin': False}
    r = await async_requests.get(public_url(app, mockservice_url) + '/whoami/?token=no-such-token', allow_redirects=False)
    assert r.status_code == 302
    assert 'Location' in r.headers
    location = r.headers['Location']
    path = urlparse(location).path
    assert path.endswith('/hub/login')

@pytest.mark.parametrize('scopes, allowed', [(['access:services'], True), (['access:services!service=$service'], True), (['access:services!service=other-service'], False), (['access:servers!user=$service'], False)])
async def test_hubauth_service_token(request, app, mockservice_url, scopes, allowed):
    """Test HubAuthenticated service with service API tokens"""
    scopes = [scope.replace('$service', mockservice_url.name) for scope in scopes]
    token = hexlify(os.urandom(5)).decode('utf8')
    name = 'test-api-service'
    app.service_tokens[token] = name
    await app.init_api_tokens()
    orm_service = app.db.query(orm.Service).filter_by(name=name).one()
    role_name = 'test-hubauth-service-token'
    roles.create_role(app.db, {'name': role_name, 'description': 'role for test', 'scopes': scopes})
    request.addfinalizer(lambda : roles.delete_role(app.db, role_name))
    roles.grant_role(app.db, orm_service, role_name)
    r = await async_requests.get(public_url(app, mockservice_url) + 'whoami/', headers={'Authorization': 'token %s' % token}, allow_redirects=False)
    service_model = {'kind': 'service', 'name': name, 'admin': False, 'scopes': scopes}
    if allowed:
        r.raise_for_status()
        assert r.status_code == 200
        reply = r.json()
        assert service_model.items() <= reply.items()
        assert not r.cookies
    else:
        assert r.status_code == 403
    r = await async_requests.get(public_url(app, mockservice_url) + 'whoami/?token=%s' % token)
    if allowed:
        r.raise_for_status()
        assert r.status_code == 200
        reply = r.json()
        assert service_model.items() <= reply.items()
        assert not r.cookies
    else:
        assert r.status_code == 403
    r = await async_requests.get(public_url(app, mockservice_url) + 'whoami/?token=no-such-token', allow_redirects=False)
    assert r.status_code == 302
    assert 'Location' in r.headers
    location = r.headers['Location']
    path = urlparse(location).path
    assert path.endswith('/hub/login')

@pytest.mark.parametrize('client_allowed_roles, request_scopes, expected_scopes', [([], [], []), ([], ['identify'], []), ([], ['admin'], None), ([], ['read:users'], None), ([], ['nosuchscope'], None), ([], ['admin:invalid!no=bo!'], None), (['user'], ['user'], ['user']), (['token', 'user'], [], ['token', 'user']), (['token', 'user'], ['identify'], ['read:users:name!user=$user']), (['token', 'server'], ['token', 'user'], None), (['read-only'], ['access:services'], None), (['admin', 'user'], ['user'], ['user']), (['user', 'token', 'server'], ['token', 'user'], ['token', 'user']), (['admin', 'user', 'read-only'], ['read-only'], ['read-only']), (['read-only'], ['access:servers'], ['access:servers']), (['admin', 'user'], ['admin:users', 'access:servers', 'self'], ['access:servers', 'user']), (['other'], ['other'], []), (['user'], ['custom:jupyter_server:read:*'], None), (['read-only'], ['custom:jupyter_server:read:*'], ['custom:jupyter_server:read:*']), (['read-only'], ['custom:jupyter_server:read:*!user=$user'], ['custom:jupyter_server:read:*!user=$user'])])
async def test_oauth_service_roles(app, mockservice_url, create_user_with_scopes, client_allowed_roles, request_scopes, expected_scopes, preserve_scopes):
    service = mockservice_url
    oauth_client = app.db.query(orm.OAuthClient).filter_by(identifier=service.oauth_client_id).one()
    scopes.define_custom_scopes({'custom:jupyter_server:read:*': {'description': 'read-only access to jupyter server'}})
    roles.create_role(app.db, {'name': 'read-only', 'description': 'read-only access to servers', 'scopes': ['access:servers', 'custom:jupyter_server:read:*']})
    roles.create_role(app.db, {'name': 'other', 'description': 'A role not held by our test user', 'scopes': ['admin-ui']})
    oauth_client.allowed_scopes = sorted(roles_to_scopes([orm.Role.find(app.db, role_name) for role_name in client_allowed_roles]))
    app.db.commit()
    user = create_user_with_scopes('access:services')
    url = url_path_join(public_url(app, mockservice_url) + 'owhoami/?arg=x')
    if request_scopes:
        request_scopes = {s.replace('$user', user.name) for s in request_scopes}
        url = url_concat(url, {'request-scope': ' '.join(request_scopes)})
    s = AsyncSession()
    roles.grant_role(app.db, user, 'user')
    roles.grant_role(app.db, user, 'read-only')
    name = user.name
    s.cookies = await app.login_user(name)
    r = await s.get(url)
    if expected_scopes is None:
        (dest_url, _, query) = r.url.partition('?')
        assert dest_url == public_url(app, mockservice_url) + 'oauth_callback'
        assert parse_qs(query).get('error') == ['invalid_scope']
        assert r.status_code == 400
        return
    r.raise_for_status()
    assert urlparse(r.url).path == app.base_url + 'hub/api/oauth2/authorize'
    assert set(r.history[0].cookies.keys()) == {'service-%s-oauth-state' % service.name}
    page = BeautifulSoup(r.text, 'html.parser')
    scope_inputs = page.find_all('input', {'name': 'scopes'})
    scope_values = [input['value'] for input in scope_inputs]
    app_log.info(f'Submitting request with scope values {scope_values}')
    data = {}
    if scope_values:
        data['scopes'] = scope_values
    data['_xsrf'] = s.cookies['_xsrf']
    r = await s.post(r.url, data=data)
    r.raise_for_status()
    assert r.url == url
    assert 'service-%s' % service.name in set(s.cookies.keys())
    assert 'service-%s-oauth-state' % service.name not in set(s.cookies.keys())
    r = await s.get(url, allow_redirects=False)
    r.raise_for_status()
    assert r.status_code == 200
    assert len(r.history) == 0
    reply = r.json()
    sub_reply = {key: reply.get(key, 'missing') for key in ('kind', 'name')}
    assert sub_reply == {'name': user.name, 'kind': 'user'}
    expected_scopes = {s.replace('$user', user.name) for s in expected_scopes}
    for scope in list(expected_scopes):
        role = orm.Role.find(app.db, scope)
        if role:
            expected_scopes.discard(role.name)
            expected_scopes.update(roles.roles_to_expanded_scopes([role], owner=user.orm_user))
    if 'inherit' in expected_scopes:
        expected_scopes = set(scopes.get_scopes_for(user.orm_user))
    expected_scopes.update(scopes.identify_scopes(user.orm_user))
    expected_scopes.update(scopes.access_scopes(oauth_client))
    expected_scopes = scopes.reduce_scopes(expected_scopes)
    have_scopes = scopes.reduce_scopes(set(reply['scopes']))
    assert sorted(have_scopes) == sorted(expected_scopes)
    token = app.users[name].new_api_token()
    r = await async_requests.get(url_concat(url, {'token': token}))
    r.raise_for_status()
    reply = r.json()
    assert reply['name'] == name
    assert len(r.cookies) != 0
    r = await async_requests.get(url, cookies=r.cookies, allow_redirects=False)
    r.raise_for_status()
    assert r.url == url
    reply = r.json()
    assert reply['name'] == name

@pytest.mark.parametrize('access_scopes, expect_success', [(['access:services'], True), (['access:services!service=$service'], True), (['access:services!service=other-service'], False), (['self'], False), ([], False)])
async def test_oauth_access_scopes(app, mockservice_url, create_user_with_scopes, access_scopes, expect_success):
    """Check that oauth/authorize validates access scopes"""
    service = mockservice_url
    access_scopes = [s.replace('$service', service.name) for s in access_scopes]
    url = url_path_join(public_url(app, mockservice_url) + 'owhoami/?arg=x')
    s = AsyncSession()
    user = create_user_with_scopes(*access_scopes)
    name = user.name
    s.cookies = await app.login_user(name)
    r = await s.get(url)
    if not expect_success:
        assert r.status_code == 403
        return
    r.raise_for_status()
    assert urlparse(r.url).path == app.base_url + 'hub/api/oauth2/authorize'
    assert set(r.history[0].cookies.keys()) == {'service-%s-oauth-state' % service.name}
    r = await s.post(r.url, data={'_xsrf': s.cookies['_xsrf']})
    r.raise_for_status()
    assert r.url == url
    assert 'service-%s' % service.name in set(s.cookies.keys())
    assert 'service-%s-oauth-state' % service.name not in set(s.cookies.keys())
    r = await s.get(url, allow_redirects=False)
    r.raise_for_status()
    assert r.status_code == 200
    reply = r.json()
    sub_reply = {key: reply.get(key, 'missing') for key in ('kind', 'name')}
    assert sub_reply == {'name': name, 'kind': 'user'}
    user.roles = []
    app.db.commit()
    s.cookies.pop('jupyterhub-session-id')
    r = await s.get(url, allow_redirects=False)
    assert r.status_code == 403

@pytest.mark.parametrize('token_roles, hits_page', [([], True), (['writer'], True), (['writer', 'reader'], False)])
async def test_oauth_page_hit(app, mockservice_url, create_user_with_scopes, create_temp_role, token_roles, hits_page):
    test_roles = {'reader': create_temp_role(['read:users!user'], role_name='reader'), 'writer': create_temp_role(['users:activity!user'], role_name='writer')}
    service = mockservice_url
    user = create_user_with_scopes('access:services', 'self')
    for role in test_roles.values():
        roles.grant_role(app.db, user, role)
    oauth_client = app.db.query(orm.OAuthClient).filter_by(identifier=service.oauth_client_id).one()
    oauth_client.allowed_scopes = sorted(roles_to_scopes(list(test_roles.values())))
    authorized_scopes = roles.roles_to_scopes([test_roles[t] for t in token_roles])
    authorized_scopes.update(scopes.identify_scopes())
    authorized_scopes.update(scopes.access_scopes(oauth_client))
    user.new_api_token(scopes=authorized_scopes)
    token = user.api_tokens[0]
    token.client_id = service.oauth_client_id
    app.db.commit()
    s = AsyncSession()
    s.cookies = await app.login_user(user.name)
    url = url_path_join(public_url(app, service) + 'owhoami/?arg=x')
    r = await s.get(url)
    r.raise_for_status()
    if hits_page:
        assert urlparse(r.url).path == app.base_url + 'hub/api/oauth2/authorize'
    else:
        assert r.status_code == 200
        assert r.url == url

async def test_oauth_cookie_collision(app, mockservice_url, create_user_with_scopes):
    service = mockservice_url
    url = url_path_join(public_url(app, mockservice_url), 'owhoami/')
    print(url)
    s = AsyncSession()
    name = 'mypha'
    user = create_user_with_scopes('access:services', name=name)
    s.cookies = await app.login_user(name)
    state_cookie_name = 'service-%s-oauth-state' % service.name
    service_cookie_name = 'service-%s' % service.name
    oauth_1 = await s.get(url)
    print(oauth_1.headers)
    print(oauth_1.cookies, oauth_1.url, url)
    assert state_cookie_name in s.cookies
    state_cookies = [c for c in s.cookies.keys() if c.startswith(state_cookie_name)]
    assert state_cookies == [state_cookie_name]
    state_1 = s.cookies[state_cookie_name]
    oauth_2 = await s.get(url)
    state_cookies = [c for c in s.cookies.keys() if c.startswith(state_cookie_name)]
    assert len(state_cookies) == 2
    state_cookie_2 = sorted(state_cookies)[-1]
    assert s.cookies[state_cookie_name] == state_1
    r = await s.post(oauth_2.url, data={'scopes': ['identify'], '_xsrf': s.cookies['_xsrf']})
    r.raise_for_status()
    assert r.url == url
    assert state_cookie_2 not in s.cookies
    assert service_cookie_name in s.cookies
    service_cookie_2 = s.cookies[service_cookie_name]
    r = await s.post(oauth_1.url, data={'scopes': ['identify'], '_xsrf': s.cookies['_xsrf']})
    r.raise_for_status()
    assert r.url == url
    assert state_cookie_name not in s.cookies
    assert service_cookie_name in s.cookies
    assert s.cookies[service_cookie_name] != service_cookie_2
    state_cookies = [s for s in s.cookies.keys() if s.startswith(state_cookie_name)]
    assert state_cookies == []

async def test_oauth_logout(app, mockservice_url, create_user_with_scopes):
    """Verify that logout via the Hub triggers logout for oauth services

    1. clears session id cookie
    2. revokes oauth tokens
    3. cleared session id ensures cached auth miss
    4. cache hit
    """
    service = mockservice_url
    service_cookie_name = 'service-%s' % service.name
    url = url_path_join(public_url(app, mockservice_url), 'owhoami/?foo=bar')
    s = AsyncSession()
    name = 'propha'
    user = create_user_with_scopes('access:services', name=name)

    def auth_tokens():
        if False:
            while True:
                i = 10
        'Return list of OAuth access tokens for the user'
        return list(app.db.query(orm.APIToken).filter_by(user_id=user.id))
    assert auth_tokens() == []
    s.cookies = await app.login_user(name)
    assert 'jupyterhub-session-id' in s.cookies
    r = await s.get(url)
    r.raise_for_status()
    assert urlparse(r.url).path.endswith('oauth2/authorize')
    r = await s.post(r.url, data={'scopes': ['identify'], '_xsrf': s.cookies['_xsrf']})
    r.raise_for_status()
    assert r.url == url
    r = await s.get(url, allow_redirects=False)
    r.raise_for_status()
    assert r.status_code == 200
    reply = r.json()
    sub_reply = {key: reply.get(key, 'missing') for key in ('kind', 'name')}
    assert sub_reply == {'name': name, 'kind': 'user'}
    saved_cookies = copy.deepcopy(s.cookies)
    session_id = s.cookies['jupyterhub-session-id']
    assert len(auth_tokens()) == 1
    token = auth_tokens()[0]
    assert token.expires_in is not None
    assert abs(app.oauth_token_expires_in - token.expires_in) < 30
    r = await s.get(public_url(app, path='hub/logout'))
    r.raise_for_status()
    assert sorted(s.cookies.keys()) == ['_xsrf', service_cookie_name]
    r = await s.get(url)
    r.raise_for_status()
    assert r.url.split('?')[0] == public_url(app, path='hub/login')
    assert auth_tokens() == []
    s.cookies = saved_cookies
    assert session_id == s.cookies['jupyterhub-session-id']
    r = await s.get(url, allow_redirects=False)
    r.raise_for_status()
    assert r.status_code == 200
    reply = r.json()
    sub_reply = {key: reply.get(key, 'missing') for key in ('kind', 'name')}
    assert sub_reply == {'name': name, 'kind': 'user'}