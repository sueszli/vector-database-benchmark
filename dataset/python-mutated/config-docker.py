import os

def env_to_bool(input):
    if False:
        print('Hello World!')
    '\n        Must change String from environment variable into Boolean\n        defaults to True\n    '
    if isinstance(input, str):
        if input in ('False', 'false'):
            return False
        else:
            return True
    else:
        return input
LOG_CFG = {'version': 1, 'disable_existing_loggers': False, 'formatters': {'standard': {'format': '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'}}, 'handlers': {'file': {'class': 'logging.handlers.GroupWriteRotatingFileHandler', 'level': 'DEBUG', 'formatter': 'standard', 'filename': '/var/log/security_monkey/securitymonkey.log', 'maxBytes': 10485760, 'backupCount': 100, 'encoding': 'utf8'}, 'console': {'class': 'logging.StreamHandler', 'level': 'DEBUG', 'formatter': 'standard', 'stream': 'ext://sys.stdout'}}, 'loggers': {'security_monkey': {'handlers': ['console'], 'level': os.getenv('SM_CONSOLE_LOG_LEVEL', 'DEBUG')}, 'apscheduler': {'handlers': ['console'], 'level': os.getenv('SM_APPSCHEDULER_LOG_LEVEL', 'INFO')}}}
SQLALCHEMY_DATABASE_URI = 'postgresql://%s:%s@%s:%s/%s' % (os.getenv('SECURITY_MONKEY_POSTGRES_USER', 'postgres'), os.getenv('SECURITY_MONKEY_POSTGRES_PASSWORD', 'securitymonkeypassword'), os.getenv('SECURITY_MONKEY_POSTGRES_HOST', 'localhost'), os.getenv('SECURITY_MONKEY_POSTGRES_PORT', 5432), os.getenv('SECURITY_MONKEY_POSTGRES_DATABASE', 'secmonkey'))
ENVIRONMENT = 'ec2'
USE_ROUTE53 = False
FQDN = os.getenv('SECURITY_MONKEY_FQDN', 'ec2-XX-XXX-XXX-XXX.compute-1.amazonaws.com')
API_PORT = '5000'
WEB_PORT = '443'
WEB_PATH = '/static/ui.html'
FRONTED_BY_NGINX = True
NGINX_PORT = '443'
BASE_URL = 'https://{}/'.format(FQDN)
SECRET_KEY = os.getenv('SECURITY_MONKEY_SECRET_KEY', '<INSERT_RANDOM_STRING_HERE>')
MAIL_DEFAULT_SENDER = os.getenv('SECURITY_MONKEY_EMAIL_DEFAULT_SENDER', 'securitymonkey@example.com')
SECURITY_REGISTERABLE = env_to_bool(os.getenv('SECURITY_MONKEY_SECURITY_REGISTERABLE', False))
SECURITY_CONFIRMABLE = env_to_bool(os.getenv('SECURITY_MONKEY_SECURITY_CONFIRMABLE', False))
SECURITY_RECOVERABLE = env_to_bool(os.getenv('SECURITY_MONKEY_SECURITY_RECOVERABLE', False))
SECURITY_PASSWORD_HASH = 'bcrypt'
SECURITY_PASSWORD_SALT = os.getenv('SECURITY_MONKEY_SECURITY_PASSWORD_SALT', '<INSERT_RANDOM_STRING_HERE>')
SECURITY_TRACKABLE = True
SECURITY_POST_LOGIN_VIEW = BASE_URL
SECURITY_POST_REGISTER_VIEW = BASE_URL
SECURITY_POST_CONFIRM_VIEW = BASE_URL
SECURITY_POST_RESET_VIEW = BASE_URL
SECURITY_POST_CHANGE_VIEW = BASE_URL
SECURITY_TEAM_EMAIL = os.getenv('SECURITY_MONKEY_SECURITY_TEAM_EMAIL', [])
EMAIL_AUDIT_REPORTS_INCLUDE_JUSTIFIED = env_to_bool(os.getenv('SECURITY_MONKEY_EMAIL_AUDIT_REPORTS_INCLUDE_JUSTIFIED', True))
EMAILS_USE_SMTP = env_to_bool(os.getenv('SECURITY_MONKEY_SMTP', True))
SES_REGION = os.getenv('SECURITY_MONKEY_SES_REGION', 'us-east-1')
MAIL_SERVER = os.getenv('SECURITY_MONKEY_EMAIL_SERVER', 'smtp.example.com')
MAIL_PORT = 465
MAIL_USE_SSL = True
MAIL_USERNAME = os.getenv('SECURITY_MONKEY_EMAIL_USERNAME', 'username')
MAIL_PASSWORD = os.getenv('SECURITY_MONKEY_EMAIL_PASSWORD', 'password')
WTF_CSRF_ENABLED = env_to_bool(os.getenv('SM_WTF_CSRF_ENABLED', True))
WTF_CSRF_SSL_STRICT = env_to_bool(os.getenv('SM_WTF_CSRF_SSL_STRICT', True))
WTF_CSRF_METHODS = ['DELETE', 'POST', 'PUT', 'PATCH']
SECURITYGROUP_INSTANCE_DETAIL = 'FULL'
ACTIVE_PROVIDERS = []
if os.getenv('SECURITY_MONKEY_ACTIVE_PROVIDERS'):
    ACTIVE_PROVIDERS = [os.getenv('SECURITY_MONKEY_ACTIVE_PROVIDERS')]
PING_NAME = ''
PING_REDIRECT_URI = '{BASE}api/1/auth/ping'.format(BASE=BASE_URL)
PING_CLIENT_ID = ''
PING_AUTH_ENDPOINT = ''
PING_ACCESS_TOKEN_URL = ''
PING_USER_API_URL = ''
PING_JWKS_URL = ''
PING_SECRET = ''
GOOGLE_CLIENT_ID = os.getenv('SECURITY_MONKEY_GOOGLE_CLIENT_ID', '')
GOOGLE_AUTH_ENDPOINT = os.getenv('SECURITY_MONKEY_GOOGLE_AUTH_ENDPOINT', '')
GOOGLE_SECRET = os.getenv('SECURITY_MONKEY_GOOGLE_SECRET', '')
GOOGLE_HOSTED_DOMAIN = os.getenv('SECURITY_MONKEY_GOOGLE_HOSTED_DOMAIN', '')
GOOGLE_DOMAIN_WIDE_DELEGATION_KEY_PATH = os.getenv('SECURITY_MONKEY_GOOGLE_DOMAIN_WIDE_DELEGATION_KEY_PATH', '')
GOOGLE_DOMAIN_WIDE_DELEGATION_KEY_JSON = os.getenv('SECURITY_MONKEY_GOOGLE_DOMAIN_WIDE_DELEGATION_KEY_JSON', '')
GOOGLE_ADMIN_ROLE_GROUP_NAME = ''
GOOGLE_AUTH_API_METHOD = 'People'
GOOGLE_DOMAIN_WIDE_DELEGATION_SUBJECT = ''
OKTA_NAME = os.getenv('SECURITY_MONKEY_OKTA_NAME', 'Okta')
OKTA_AUTH_SERVER = os.getenv('SECURITY_MONKEY_OKTA_AUTH_SERVER', 'default')
OKTA_BASE_URL = os.getenv('SECURITY_MONKEY_OKTA_BASE_URL', '')
OKTA_AUTH_ENDPOINT = '{OKTA_BASE}/oauth2/{AUTH_SERVER}/v1/authorize'.format(OKTA_BASE=OKTA_BASE_URL, AUTH_SERVER=OKTA_AUTH_SERVER)
OKTA_TOKEN_ENDPOINT = '{OKTA_BASE}/oauth2/{AUTH_SERVER}/v1/token'.format(OKTA_BASE=OKTA_BASE_URL, AUTH_SERVER=OKTA_AUTH_SERVER)
OKTA_USER_INFO_ENDPOINT = '{OKTA_BASE}/oauth2/{AUTH_SERVER}/v1/userinfo'.format(OKTA_BASE=OKTA_BASE_URL, AUTH_SERVER=OKTA_AUTH_SERVER)
OKTA_JWKS_URI = '{OKTA_BASE}/oauth2/{AUTH_SERVER}/v1/keys'.format(OKTA_BASE=OKTA_BASE_URL, AUTH_SERVER=OKTA_AUTH_SERVER)
OKTA_CLIENT_ID = os.getenv('SECURITY_MONKEY_OKTA_CLIENT_ID', '')
OKTA_CLIENT_SECRET = os.getenv('SECURITY_MONKEY_OKTA_CLIENT_SECRET', '')
OKTA_REDIRECT_URI = '{BASE}/api/1/auth/okta'.format(BASE=BASE_URL)
OKTA_DEFAULT_ROLE = os.getenv('SECURITY_MONKEY_OKTA_DEFAULT_ROLE', 'View')
ONELOGIN_APP_ID = os.getenv('SECURITY_MONKEY_ONELOGIN_APP_ID', '<APP_ID>')
ONELOGIN_EMAIL_FIELD = os.getenv('SECURITY_MONKEY_ONELOGIN_EMAIL_FIELD', 'User.email')
ONELOGIN_DEFAULT_ROLE = 'View'
ONELOGIN_HTTPS = True
ONELOGIN_IDP_CERT = os.getenv('SECURITY_MONKEY_ONELOGIN_IDP_CERT', '<IDP_CERT>')
ONELOGIN_USE_CUSTOM = os.getenv('SECURITY_MONKEY_ONELOGIN_USE_CUSTOM', False)
if not ONELOGIN_USE_CUSTOM:
    ONELOGIN_ENTITY_ID = 'https://app.onelogin.com/saml/metadata/{APP_ID}'.format(APP_ID=ONELOGIN_APP_ID)
    ONELOGIN_SSO_URL = 'https://app.onelogin.com/trust/saml2/http-post/sso/{APP_ID}'.format(APP_ID=ONELOGIN_APP_ID)
    ONELOGIN_SLO_URL = 'https://app.onelogin.com/trust/saml2/http-redirect/slo/{APP_ID}'.format(APP_ID=ONELOGIN_APP_ID)
else:
    ONELOGIN_ENTITY_ID = os.getenv('SECURITY_MONKEY_ONELOGIN_ENTITY_ID')
    ONELOGIN_SSO_URL = os.getenv('SECURITY_MONKEY_ONELOGIN_SSO_URL')
    ONELOGIN_SLO_URL = os.getenv('SECURITY_MONKEY_ONELOGIN_SLO_URL')
ONELOGIN_SETTINGS = {'strict': True, 'debug': True, 'sp': {'entityId': '{BASE}metadata/'.format(BASE=BASE_URL), 'assertionConsumerService': {'url': '{BASE}api/1/auth/onelogin?acs'.format(BASE=BASE_URL), 'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST'}, 'singleLogoutService': {'url': '{BASE}api/1/auth/onelogin?sls'.format(BASE=BASE_URL), 'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'}, 'NameIDFormat': 'urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified', 'x509cert': '', 'privateKey': ''}, 'idp': {'entityId': ONELOGIN_ENTITY_ID, 'singleSignOnService': {'url': ONELOGIN_SSO_URL, 'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'}, 'singleLogoutService': {'url': ONELOGIN_SLO_URL, 'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'}, 'x509cert': ONELOGIN_IDP_CERT}}
from datetime import timedelta
PERMANENT_SESSION_LIFETIME = timedelta(minutes=60)
SESSION_REFRESH_EACH_REQUEST = True
SESSION_COOKIE_SECURE = env_to_bool(os.getenv('SESSION_COOKIE_SECURE', True))
SESSION_COOKIE_HTTPONLY = True
PREFERRED_URL_SCHEME = 'https'
REMEMBER_COOKIE_DURATION = timedelta(minutes=60)
REMEMBER_COOKIE_SECURE = True
REMEMBER_COOKIE_HTTPONLY = True
USE_HEADER_AUTH = env_to_bool(os.getenv('SECURITY_MONKEY_USE_HEADER_AUTH', False))
HEADER_AUTH_USERNAME_HEADER = os.getenv('SECURITY_MONKEY_HEADER_AUTH_USERNAME_HEADER', 'Remote-User')
HEADER_AUTH_GROUPS_HEADER = os.getenv('SECURITY_MONKEY_HEADER_AUTH_GROUPS_HEADER')
LOG_SSL_SUBJ_ALT_NAME_ERRORS = True