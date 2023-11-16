import os
from typing import Dict, List, Optional, Tuple
import ldap
from django_auth_ldap.config import LDAPSearch
from zerver.lib.db import TimeTrackingConnection, TimeTrackingCursor
from zproject.settings_types import OIDCIdPConfigDict, SAMLIdPConfigDict, SCIMConfigDict
from .config import DEPLOY_ROOT, get_from_file_if_exists
from .settings import AUTHENTICATION_BACKENDS, CACHES, DATABASES, EXTERNAL_HOST, LOCAL_DATABASE_PASSWORD, LOGGING, MIDDLEWARE
FULL_STACK_ZULIP_TEST = 'FULL_STACK_ZULIP_TEST' in os.environ
PUPPETEER_TESTS = 'PUPPETEER_TESTS' in os.environ
FAKE_EMAIL_DOMAIN = 'zulip.testserver'
REALM_HOSTS: Dict[str, str] = {}
DATABASES['default'] = {'NAME': os.getenv('ZULIP_DB_NAME', 'zulip_test'), 'USER': 'zulip_test', 'PASSWORD': LOCAL_DATABASE_PASSWORD, 'HOST': 'localhost', 'SCHEMA': 'zulip', 'ENGINE': 'django.db.backends.postgresql', 'TEST_NAME': 'django_zulip_tests', 'OPTIONS': {'connection_factory': TimeTrackingConnection, 'cursor_factory': TimeTrackingCursor}}
if FULL_STACK_ZULIP_TEST:
    TORNADO_PORTS = [9983]
else:
    USING_TORNADO = False
    CAMO_URI = 'https://external-content.zulipcdn.net/external_content/'
    CAMO_KEY = 'dummy'
if 'RUNNING_OPENAPI_CURL_TEST' in os.environ:
    RUNNING_OPENAPI_CURL_TEST = True
if 'GENERATE_STRIPE_FIXTURES' in os.environ:
    GENERATE_STRIPE_FIXTURES = True
if 'BAN_CONSOLE_OUTPUT' in os.environ:
    BAN_CONSOLE_OUTPUT = True
EVENT_QUEUE_LONGPOLL_TIMEOUT_SECONDS = 1
EMAIL_BACKEND = 'django.core.mail.backends.locmem.EmailBackend'
AUTHENTICATION_BACKENDS += ('zproject.backends.EmailAuthBackend',)
GOOGLE_OAUTH2_CLIENT_ID = 'test_client_id'
AUTH_LDAP_ALWAYS_UPDATE_USER = False
AUTH_LDAP_USER_SEARCH = LDAPSearch('ou=users,dc=zulip,dc=com', ldap.SCOPE_ONELEVEL, '(uid=%(user)s)')
AUTH_LDAP_USERNAME_ATTR = 'uid'
AUTH_LDAP_REVERSE_EMAIL_SEARCH = LDAPSearch('ou=users,dc=zulip,dc=com', ldap.SCOPE_ONELEVEL, '(mail=%(email)s)')
TEST_SUITE = True
RATE_LIMITING = False
RATE_LIMITING_AUTHENTICATE = False
USING_RABBITMQ = False
CACHES['database'] = {'BACKEND': 'django.core.cache.backends.dummy.DummyCache', 'LOCATION': 'zulip-database-test-cache', 'TIMEOUT': 3600, 'CONN_MAX_AGE': 600, 'OPTIONS': {'MAX_ENTRIES': 100000}}
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
if PUPPETEER_TESTS:
    WEBPACK_STATS_FILE = os.path.join(DEPLOY_ROOT, 'webpack-stats-production.json')
else:
    WEBPACK_STATS_FILE = os.path.join(DEPLOY_ROOT, 'var', 'webpack-stats-test.json')
WEBPACK_BUNDLES = 'webpack-bundles/'
if not PUPPETEER_TESTS:
    CACHES['default'] = {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}

    def set_loglevel(logger_name: str, level: str) -> None:
        if False:
            print('Hello World!')
        LOGGING['loggers'].setdefault(logger_name, {})['level'] = level
    set_loglevel('zulip.requests', 'CRITICAL')
    set_loglevel('zulip.management', 'CRITICAL')
    set_loglevel('zulip.auth', 'WARNING')
    set_loglevel('django.request', 'ERROR')
    set_loglevel('django_auth_ldap', 'WARNING')
    set_loglevel('fakeldap', 'ERROR')
    set_loglevel('zulip.send_email', 'ERROR')
    set_loglevel('zerver.lib.push_notifications', 'WARNING')
    set_loglevel('zerver.lib.digest', 'ERROR')
    set_loglevel('zerver.lib.email_mirror', 'ERROR')
    set_loglevel('zerver.worker.queue_processors', 'WARNING')
    set_loglevel('stripe', 'WARNING')
ENABLE_FILE_LINKS = True
if 'LOCAL_UPLOADS_DIR' in os.environ:
    LOCAL_UPLOADS_DIR = os.getenv('LOCAL_UPLOADS_DIR')
    assert LOCAL_UPLOADS_DIR is not None
    LOCAL_AVATARS_DIR = os.path.join(LOCAL_UPLOADS_DIR, 'avatars')
    LOCAL_FILES_DIR = os.path.join(LOCAL_UPLOADS_DIR, 'files')
S3_KEY = 'test-key'
S3_SECRET_KEY = 'test-secret-key'
S3_AUTH_UPLOADS_BUCKET = 'test-authed-bucket'
S3_AVATAR_BUCKET = 'test-avatar-bucket'
INLINE_URL_EMBED_PREVIEW = False
HOME_NOT_LOGGED_IN = '/login/'
LOGIN_URL = '/accounts/login/'
APNS_CERT_FILE: Optional[str] = None
SEND_LOGIN_EMAILS = False
GOOGLE_OAUTH2_CLIENT_ID = 'id'
GOOGLE_OAUTH2_CLIENT_SECRET = 'secret'
SOCIAL_AUTH_GITHUB_KEY = 'key'
SOCIAL_AUTH_GITHUB_SECRET = 'secret'
SOCIAL_AUTH_GITLAB_KEY = 'key'
SOCIAL_AUTH_GITLAB_SECRET = 'secret'
SOCIAL_AUTH_GOOGLE_KEY = 'key'
SOCIAL_AUTH_GOOGLE_SECRET = 'secret'
SOCIAL_AUTH_SUBDOMAIN = 'auth'
SOCIAL_AUTH_APPLE_SERVICES_ID = 'com.zulip.chat'
SOCIAL_AUTH_APPLE_APP_ID = 'com.zulip.bundle.id'
SOCIAL_AUTH_APPLE_CLIENT = 'com.zulip.chat'
SOCIAL_AUTH_APPLE_AUDIENCE = [SOCIAL_AUTH_APPLE_APP_ID, SOCIAL_AUTH_APPLE_SERVICES_ID]
SOCIAL_AUTH_APPLE_KEY = 'KEYISKEY'
SOCIAL_AUTH_APPLE_TEAM = 'TEAMSTRING'
SOCIAL_AUTH_APPLE_SECRET = get_from_file_if_exists('zerver/tests/fixtures/apple/private_key.pem')
SOCIAL_AUTH_OIDC_ENABLED_IDPS: Dict[str, OIDCIdPConfigDict] = {'testoidc': {'display_name': 'Test OIDC', 'oidc_url': 'https://example.com/api/openid', 'display_icon': None, 'client_id': 'key', 'secret': 'secret'}}
SOCIAL_AUTH_OIDC_FULL_NAME_VALIDATED = True
VIDEO_ZOOM_CLIENT_ID = 'client_id'
VIDEO_ZOOM_CLIENT_SECRET = 'client_secret'
BIG_BLUE_BUTTON_SECRET = '123'
BIG_BLUE_BUTTON_URL = 'https://bbb.example.com/bigbluebutton/'
TWO_FACTOR_AUTHENTICATION_ENABLED = False
PUSH_NOTIFICATION_BOUNCER_URL: Optional[str] = None
DEVELOPMENT_LOG_EMAILS = False
SOCIAL_AUTH_SAML_SP_ENTITY_ID = 'http://' + EXTERNAL_HOST
SOCIAL_AUTH_SAML_SP_PUBLIC_CERT = get_from_file_if_exists('zerver/tests/fixtures/saml/zulip.crt')
SOCIAL_AUTH_SAML_SP_PRIVATE_KEY = get_from_file_if_exists('zerver/tests/fixtures/saml/zulip.key')
SOCIAL_AUTH_SAML_ORG_INFO = {'en-US': {'name': 'example', 'displayname': 'Example Inc.', 'url': '{}{}'.format('http://', EXTERNAL_HOST)}}
SOCIAL_AUTH_SAML_TECHNICAL_CONTACT = {'givenName': 'Tech Gal', 'emailAddress': 'technical@example.com'}
SOCIAL_AUTH_SAML_SUPPORT_CONTACT = {'givenName': 'Support Guy', 'emailAddress': 'support@example.com'}
SOCIAL_AUTH_SAML_ENABLED_IDPS: Dict[str, SAMLIdPConfigDict] = {'test_idp': {'entity_id': 'https://idp.testshib.org/idp/shibboleth', 'url': 'https://idp.testshib.org/idp/profile/SAML2/Redirect/SSO', 'slo_url': 'https://idp.testshib.org/idp/profile/SAML2/Redirect/Logout', 'sp_initiated_logout_enabled': True, 'x509cert': get_from_file_if_exists('zerver/tests/fixtures/saml/idp.crt'), 'attr_user_permanent_id': 'email', 'attr_first_name': 'first_name', 'attr_last_name': 'last_name', 'attr_username': 'email', 'attr_email': 'email', 'display_name': 'Test IdP'}}
RATE_LIMITING_RULES: Dict[str, List[Tuple[int, int]]] = {'api_by_user': [], 'api_by_ip': [], 'api_by_remote_server': [], 'authenticate_by_username': [], 'sends_email_by_ip': [], 'email_change_by_user': [], 'password_reset_form_by_email': []}
FREE_TRIAL_DAYS: Optional[int] = None
SCIM_CONFIG: Dict[str, SCIMConfigDict] = {'zulip': {'bearer_token': 'token1234', 'scim_client_name': 'test-scim-client', 'name_formatted_included': True}}
while len(MIDDLEWARE) < 19:
    MIDDLEWARE += ['zerver.middleware.ZulipNoopMiddleware']