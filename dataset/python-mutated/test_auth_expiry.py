"""
test authentication expiry

authentication can expire in a number of ways:

- needs refresh and can be refreshed
- doesn't need refresh
- needs refresh and cannot be refreshed without new login
"""
from unittest import mock
from urllib.parse import parse_qs, urlparse
import pytest
from .utils import api_request, get_page

async def refresh_expired(authenticator, user):
    return None

@pytest.fixture
def disable_refresh(app):
    if False:
        i = 10
        return i + 15
    'Fixture disabling auth refresh'
    with mock.patch.object(app.authenticator, 'refresh_user', refresh_expired):
        yield

@pytest.fixture
def refresh_pre_spawn(app):
    if False:
        for i in range(10):
            print('nop')
    'Fixture enabling auth refresh pre spawn'
    app.authenticator.refresh_pre_spawn = True
    try:
        yield
    finally:
        app.authenticator.refresh_pre_spawn = False

async def test_auth_refresh_at_login(app, user):
    assert not user._auth_refreshed
    await app.login_user(user.name)
    assert user._auth_refreshed
    user._auth_refreshed -= 10
    before = user._auth_refreshed
    await app.login_user(user.name)
    assert user._auth_refreshed > before

async def test_auth_refresh_page(app, user):
    cookies = await app.login_user(user.name)
    assert user._auth_refreshed
    user._auth_refreshed -= 10
    before = user._auth_refreshed
    r = await get_page('home', app, cookies=cookies)
    assert r.status_code == 200
    assert user._auth_refreshed == before
    user._auth_refreshed -= app.authenticator.auth_refresh_age
    r = await get_page('home', app, cookies=cookies)
    assert r.status_code == 200
    assert user._auth_refreshed > before

async def test_auth_expired_page(app, user, disable_refresh):
    cookies = await app.login_user(user.name)
    assert user._auth_refreshed
    user._auth_refreshed -= 10
    before = user._auth_refreshed
    r = await get_page('home', app, cookies=cookies)
    assert user._auth_refreshed == before
    assert r.status_code == 200
    user._auth_refreshed -= app.authenticator.auth_refresh_age
    before = user._auth_refreshed
    r = await get_page('home', app, cookies=cookies, allow_redirects=False)
    assert r.status_code == 302
    redirect_url = urlparse(r.headers['Location'])
    assert redirect_url.path.endswith('/login')
    query = parse_qs(redirect_url.query)
    assert query['next']
    next_url = urlparse(query['next'][0])
    assert next_url.path == urlparse(r.url).path
    assert user._auth_refreshed == before

async def test_auth_expired_api(app, user, disable_refresh):
    cookies = await app.login_user(user.name)
    assert user._auth_refreshed
    user._auth_refreshed -= 10
    before = user._auth_refreshed
    r = await api_request(app, 'users/' + user.name, name=user.name)
    assert user._auth_refreshed == before
    assert r.status_code == 200
    user._auth_refreshed -= app.authenticator.auth_refresh_age
    r = await api_request(app, 'users/' + user.name, name=user.name)
    assert r.status_code == 403

async def test_refresh_pre_spawn(app, user, refresh_pre_spawn):
    cookies = await app.login_user(user.name)
    assert user._auth_refreshed
    user._auth_refreshed -= 10
    before = user._auth_refreshed
    r = await api_request(app, f'users/{user.name}/server', method='post', name=user.name)
    assert 200 <= r.status_code < 300
    assert user._auth_refreshed > before

async def test_refresh_pre_spawn_expired(app, user, refresh_pre_spawn, disable_refresh):
    cookies = await app.login_user(user.name)
    assert user._auth_refreshed
    user._auth_refreshed -= 10
    before = user._auth_refreshed
    r = await api_request(app, f'users/{user.name}/server', method='post', name=user.name)
    assert r.status_code == 403
    assert user._auth_refreshed == before

async def test_refresh_pre_spawn_admin_request(app, user, admin_user, refresh_pre_spawn):
    await app.login_user(user.name)
    await app.login_user(admin_user.name)
    user._auth_refreshed -= 10
    before = user._auth_refreshed
    r = await api_request(app, 'users', user.name, 'server', method='post', name=admin_user.name)
    assert 200 <= r.status_code < 300
    assert user._auth_refreshed > before

async def test_refresh_pre_spawn_expired_admin_request(app, user, admin_user, refresh_pre_spawn, disable_refresh):
    await app.login_user(user.name)
    await app.login_user(admin_user.name)
    user._auth_refreshed -= 10
    user._auth_refreshed -= app.authenticator.auth_refresh_age
    r = await api_request(app, 'users', user.name, 'server', method='post', name=admin_user.name)
    assert r.status_code == 403