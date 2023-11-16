import json
from datetime import datetime, timedelta, timezone
from flask import Response, url_for
from flask.sessions import session_json_serializer
from itsdangerous import URLSafeTimedSerializer
from redis import Redis
from tests.utils import login_journalist
from tests.utils.api_helper import get_api_headers
from two_factor import TOTP
redis = Redis()
NEW_PASSWORD = 'another correct horse battery staple generic passphrase'

def _check_sig(session_cookie, journalist_app, api=False):
    if False:
        while True:
            i = 10
    if api:
        salt = 'api_' + journalist_app.config['SESSION_SIGNER_SALT']
    else:
        salt = journalist_app.config['SESSION_SIGNER_SALT']
    signer = URLSafeTimedSerializer(journalist_app.secret_key, salt)
    return signer.loads(session_cookie)

def _get_session(sid, journalist_app, api=False):
    if False:
        for i in range(10):
            print('nop')
    if api:
        key = 'api_' + journalist_app.config['SESSION_KEY_PREFIX'] + sid
    else:
        key = journalist_app.config['SESSION_KEY_PREFIX'] + sid
    return session_json_serializer.loads(redis.get(key))

def _session_from_cookiejar(cookie_jar, journalist_app):
    if False:
        i = 10
        return i + 15
    return next((cookie for cookie in cookie_jar if cookie.name == journalist_app.config['SESSION_COOKIE_NAME']), None)

def test_session_login(journalist_app, test_journo):
    if False:
        while True:
            i = 10
    with journalist_app.test_client() as app:
        login_journalist(app, test_journo['username'], test_journo['password'], test_journo['otp_secret'])
        session_cookie = _session_from_cookiejar(app.cookie_jar, journalist_app)
        assert session_cookie is not None
        sid = _check_sig(session_cookie.value, journalist_app)
        redis_session = _get_session(sid, journalist_app)
        ttl = redis.ttl(journalist_app.config['SESSION_KEY_PREFIX'] + sid)
        assert journalist_app.config['SESSION_LIFETIME'] - 10 < ttl <= journalist_app.config['SESSION_LIFETIME']
        assert redis_session['uid'] == test_journo['id']
        resp = app.get(url_for('main.index'))
        assert resp.status_code == 200

def test_session_renew(journalist_app, test_journo):
    if False:
        print('Hello World!')
    with journalist_app.test_client() as app:
        login_journalist(app, test_journo['username'], test_journo['password'], test_journo['otp_secret'])
        session_cookie = _session_from_cookiejar(app.cookie_jar, journalist_app)
        assert session_cookie is not None
        sid = _check_sig(session_cookie.value, journalist_app)
        redis_session = _get_session(sid, journalist_app)
        assert redis_session['renew_count'] == journalist_app.config['SESSION_RENEW_COUNT']
        redis.setex(name=journalist_app.config['SESSION_KEY_PREFIX'] + sid, value=session_json_serializer.dumps(redis_session), time=15 * 60)
        resp = app.get(url_for('main.index'))
        assert resp.status_code == 200
        redis_session = _get_session(sid, journalist_app)
        assert redis_session['renew_count'] == journalist_app.config['SESSION_RENEW_COUNT'] - 1
        ttl = redis.ttl(journalist_app.config['SESSION_KEY_PREFIX'] + sid)
        assert ttl > journalist_app.config['SESSION_LIFETIME']

def test_session_logout(journalist_app, test_journo):
    if False:
        while True:
            i = 10
    with journalist_app.test_client() as app:
        login_journalist(app, test_journo['username'], test_journo['password'], test_journo['otp_secret'])
        session_cookie = _session_from_cookiejar(app.cookie_jar, journalist_app)
        assert session_cookie is not None
        sid = _check_sig(session_cookie.value, journalist_app)
        assert redis.get(journalist_app.config['SESSION_KEY_PREFIX'] + sid) is not None
        resp = app.get(url_for('main.logout'), follow_redirects=False)
        assert resp.status_code == 302
        assert redis.get(journalist_app.config['SESSION_KEY_PREFIX'] + sid) is None
        resp = app.get(url_for('main.index'), follow_redirects=False)
        assert resp.status_code == 302

def test_session_admin_change_password_logout(journalist_app, test_journo, test_admin):
    if False:
        for i in range(10):
            print('nop')
    with journalist_app.test_client() as app:
        login_journalist(app, test_journo['username'], test_journo['password'], test_journo['otp_secret'])
        session_cookie = _session_from_cookiejar(app.cookie_jar, journalist_app)
        assert session_cookie is not None
        cookie_val = session_cookie.value
        sid = _check_sig(session_cookie.value, journalist_app)
        assert redis.get(journalist_app.config['SESSION_KEY_PREFIX'] + sid) is not None
    with journalist_app.test_client() as admin_app:
        login_journalist(admin_app, test_admin['username'], test_admin['password'], test_admin['otp_secret'])
        resp = admin_app.post(url_for('admin.new_password', user_id=test_journo['id']), data=dict(password=NEW_PASSWORD), follow_redirects=False)
        assert resp.status_code == 302
        assert redis.get(journalist_app.config['SESSION_KEY_PREFIX'] + sid) is None
    with journalist_app.test_client() as app:
        app.set_cookie('localhost.localdomain', 'js', cookie_val, domain='.localhost.localdomain', httponly=True, path='/')
        resp = app.get(url_for('main.index'), follow_redirects=False)
        assert resp.status_code == 302

def test_session_change_password_logout(journalist_app, test_journo):
    if False:
        for i in range(10):
            print('nop')
    with journalist_app.test_client() as app:
        login_journalist(app, test_journo['username'], test_journo['password'], test_journo['otp_secret'])
        session_cookie = _session_from_cookiejar(app.cookie_jar, journalist_app)
        assert session_cookie is not None
        sid = _check_sig(session_cookie.value, journalist_app)
        assert redis.get(journalist_app.config['SESSION_KEY_PREFIX'] + sid) is not None
        resp = app.post(url_for('account.new_password'), data=dict(current_password=test_journo['password'], token=TOTP(test_journo['otp_secret']).now(), password=NEW_PASSWORD))
        assert resp.status_code == 302
        assert redis.get(journalist_app.config['SESSION_KEY_PREFIX'] + sid) is None
        resp = app.get(url_for('main.index'), follow_redirects=False)
        assert resp.status_code == 302

def test_session_login_regenerate_sid(journalist_app, test_journo):
    if False:
        while True:
            i = 10
    with journalist_app.test_client() as app:
        resp = app.get(url_for('main.login'))
        assert resp.status_code == 200
        session_cookie_pre_login = _session_from_cookiejar(app.cookie_jar, journalist_app)
        assert session_cookie_pre_login is not None
        login_journalist(app, test_journo['username'], test_journo['password'], test_journo['otp_secret'])
        session_cookie_post_login = _session_from_cookiejar(app.cookie_jar, journalist_app)
        assert session_cookie_post_login != session_cookie_pre_login

def test_session_api_login(journalist_app, test_journo):
    if False:
        while True:
            i = 10
    with journalist_app.test_client() as app:
        resp = app.post(url_for('api.get_token'), data=json.dumps({'username': test_journo['username'], 'passphrase': test_journo['password'], 'one_time_code': TOTP(test_journo['otp_secret']).now()}), headers=get_api_headers())
        assert resp.json['journalist_uuid'] == test_journo['uuid']
        assert resp.status_code == 200
        sid = _check_sig(resp.json['token'], journalist_app, api=True)
        redis_session = _get_session(sid, journalist_app, api=True)
        assert redis_session['uid'] == test_journo['id']
        ttl = redis.ttl('api_' + journalist_app.config['SESSION_KEY_PREFIX'] + sid)
        assert journalist_app.config['SESSION_LIFETIME'] - 10 < ttl <= journalist_app.config['SESSION_LIFETIME']
        assert datetime.now(timezone.utc) < datetime.strptime(resp.json['expiration'], '%Y-%m-%dT%H:%M:%S.%f%z') < datetime.now(timezone.utc) + timedelta(seconds=journalist_app.config['SESSION_LIFETIME'])
        response = app.get(url_for('api.get_current_user'), headers=get_api_headers(resp.json['token']))
        assert response.status_code == 200
        assert response.json['uuid'] == test_journo['uuid']

def test_session_api_logout(journalist_app, test_journo):
    if False:
        print('Hello World!')
    with journalist_app.test_client() as app:
        resp = app.post(url_for('api.get_token'), data=json.dumps({'username': test_journo['username'], 'passphrase': test_journo['password'], 'one_time_code': TOTP(test_journo['otp_secret']).now()}), headers=get_api_headers())
        assert resp.json['journalist_uuid'] == test_journo['uuid']
        assert resp.status_code == 200
        token = resp.json['token']
        sid = _check_sig(token, journalist_app, api=True)
        resp = app.get(url_for('api.get_current_user'), headers=get_api_headers(token))
        assert resp.status_code == 200
        assert resp.json['uuid'] == test_journo['uuid']
        resp = app.post(url_for('api.logout'), headers=get_api_headers(token))
        assert resp.status_code == 200
        assert redis.get('api_' + journalist_app.config['SESSION_KEY_PREFIX'] + sid) is None
        resp = app.get(url_for('api.get_current_user'), headers=get_api_headers(token))
        assert resp.status_code == 403

def test_session_bad_signature(journalist_app, test_journo):
    if False:
        for i in range(10):
            print('nop')
    with journalist_app.test_client() as app:
        resp = app.post(url_for('api.get_token'), data=json.dumps({'username': test_journo['username'], 'passphrase': test_journo['password'], 'one_time_code': TOTP(test_journo['otp_secret']).now()}), headers=get_api_headers())
        assert resp.json['journalist_uuid'] == test_journo['uuid']
        assert resp.status_code == 200
        token = resp.json['token']
        sid = _check_sig(token, journalist_app, api=True)
        resp = app.get(url_for('api.get_current_user'), headers=get_api_headers(sid))
        assert resp.status_code == 403
        resp = app.get(url_for('api.get_current_user'), headers=get_api_headers(sid + '.'))
        assert resp.status_code == 403
        signer = URLSafeTimedSerializer(journalist_app.secret_key, 'wrong_salt')
        token_wrong_salt = signer.dumps(sid)
        resp = app.get(url_for('api.get_current_user'), headers=get_api_headers(token_wrong_salt))
        assert resp.status_code == 403
        signer = URLSafeTimedSerializer(journalist_app.secret_key, journalist_app.config['SESSION_SIGNER_SALT'])
        token_not_api_salt = signer.dumps(sid)
        resp = app.get(url_for('api.get_current_user'), headers=get_api_headers(token_not_api_salt))
        assert resp.status_code == 403
        resp = app.get(url_for('api.get_current_user'), headers=get_api_headers(token))
        assert resp.status_code == 200
        assert resp.json['uuid'] == test_journo['uuid']

def test_session_race_condition(mocker, journalist_app, test_journo):
    if False:
        while True:
            i = 10
    with journalist_app.test_request_context() as app:
        session = journalist_app.session_interface.open_session(journalist_app, app.request)
        assert session.sid is not None
        session['uid'] = test_journo['id']
        app.response = Response()
        journalist_app.session_interface.save_session(journalist_app, session, app.response)
        assert redis.get(journalist_app.config['SESSION_KEY_PREFIX'] + session.sid) is not None
        app.request.cookies = {journalist_app.config['SESSION_COOKIE_NAME']: session.token}
        session2 = journalist_app.session_interface.open_session(journalist_app, app.request)
        assert session2.sid == session.sid
        assert session2['uid'] == test_journo['id']
        session.modified = True
        session.new = False
        session.to_regenerate = False
        redis.delete(journalist_app.config['SESSION_KEY_PREFIX'] + session.sid)
        journalist_app.session_interface.save_session(journalist_app, session, app.response)
        assert redis.get(journalist_app.config['SESSION_KEY_PREFIX'] + session.sid) is None