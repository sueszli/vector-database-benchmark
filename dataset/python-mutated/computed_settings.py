import logging
import os
import sys
import time
from copy import deepcopy
from typing import Any, Dict, Final, List, Tuple, Union
from urllib.parse import urljoin
from scripts.lib.zulip_tools import get_tornado_ports
from zerver.lib.db import TimeTrackingConnection, TimeTrackingCursor
from .config import DEPLOY_ROOT, config_file, get_config, get_from_file_if_exists, get_mandatory_secret, get_secret
from .config import DEVELOPMENT as DEVELOPMENT
from .config import PRODUCTION as PRODUCTION
from .configured_settings import ADMINS, ALLOWED_HOSTS, AUTH_LDAP_BIND_DN, AUTH_LDAP_CONNECTION_OPTIONS, AUTH_LDAP_SERVER_URI, AUTHENTICATION_BACKENDS, CAMO_URI, CUSTOM_HOME_NOT_LOGGED_IN, DEBUG, DEBUG_ERROR_REPORTING, DEFAULT_RATE_LIMITING_RULES, EMAIL_BACKEND, EMAIL_HOST, ERROR_REPORTING, EXTERNAL_HOST, EXTERNAL_HOST_WITHOUT_PORT, EXTERNAL_URI_SCHEME, EXTRA_INSTALLED_APPS, GOOGLE_OAUTH2_CLIENT_ID, IS_DEV_DROPLET, LOCAL_UPLOADS_DIR, MEMCACHED_LOCATION, MEMCACHED_USERNAME, RATE_LIMITING_RULES, REALM_HOSTS, REGISTER_LINK_DISABLED, REMOTE_POSTGRES_HOST, REMOTE_POSTGRES_PORT, REMOTE_POSTGRES_SSLMODE, ROOT_SUBDOMAIN_ALIASES, SENTRY_DSN, SOCIAL_AUTH_APPLE_APP_ID, SOCIAL_AUTH_APPLE_SERVICES_ID, SOCIAL_AUTH_GITHUB_KEY, SOCIAL_AUTH_GITHUB_ORG_NAME, SOCIAL_AUTH_GITHUB_TEAM_ID, SOCIAL_AUTH_GOOGLE_KEY, SOCIAL_AUTH_SAML_ENABLED_IDPS, SOCIAL_AUTH_SAML_SECURITY_CONFIG, SOCIAL_AUTH_SUBDOMAIN, STATIC_URL, TORNADO_PORTS, USING_PGROONGA, ZULIP_ADMINISTRATOR
SECRET_KEY = get_mandatory_secret('secret_key')
SHARED_SECRET = get_mandatory_secret('shared_secret')
AVATAR_SALT = get_mandatory_secret('avatar_salt')
SERVER_GENERATION = int(time.time())
ZULIP_ORG_KEY = get_secret('zulip_org_key')
ZULIP_ORG_ID = get_secret('zulip_org_id')
if DEBUG:
    INTERNAL_IPS = ('127.0.0.1',)
if len(sys.argv) > 2 and sys.argv[0].endswith('manage.py') and (sys.argv[1] == 'process_queue'):
    IS_WORKER = True
else:
    IS_WORKER = False
TEST_SUITE = False
PUPPETEER_TESTS = False
RUNNING_OPENAPI_CURL_TEST = False
GENERATE_STRIPE_FIXTURES = False
BAN_CONSOLE_OUTPUT = False
TEST_WORKER_DIR = ''
REQUIRED_SETTINGS = [('EXTERNAL_HOST', 'zulip.example.com'), ('ZULIP_ADMINISTRATOR', 'zulip-admin@example.com'), ('SECRET_KEY', ''), ('AUTHENTICATION_BACKENDS', ())]
MANAGERS = ADMINS
TIME_ZONE = 'UTC'
LANGUAGE_CODE = 'en-us'
USE_I18N = True
USE_TZ = True
DEVELOPMENT_LOG_DIRECTORY = os.path.join(DEPLOY_ROOT, 'var', 'log')
ALLOWED_HOSTS += ['127.0.0.1', 'localhost']
ALLOWED_HOSTS += [EXTERNAL_HOST_WITHOUT_PORT, '.' + EXTERNAL_HOST_WITHOUT_PORT]
ALLOWED_HOSTS += REALM_HOSTS.values()
MIDDLEWARE = ['zerver.middleware.TagRequests', 'zerver.middleware.SetRemoteAddrFromRealIpHeader', 'django.contrib.sessions.middleware.SessionMiddleware', 'django.contrib.auth.middleware.AuthenticationMiddleware', 'zerver.middleware.LogRequests', 'zerver.middleware.JsonErrorHandler', 'zerver.middleware.RateLimitMiddleware', 'zerver.middleware.FlushDisplayRecipientCache', 'django.middleware.common.CommonMiddleware', 'zerver.middleware.LocaleMiddleware', 'zerver.middleware.HostDomainMiddleware', 'zerver.middleware.DetectProxyMisconfiguration', 'django.middleware.csrf.CsrfViewMiddleware', 'django_otp.middleware.OTPMiddleware', 'two_factor.middleware.threadlocals.ThreadLocals', 'zerver.middleware.FinalizeOpenGraphDescription']
AUTH_USER_MODEL = 'zerver.UserProfile'
TEST_RUNNER = 'zerver.lib.test_runner.Runner'
ROOT_URLCONF = 'zproject.urls'
WSGI_APPLICATION = 'zproject.wsgi.application'
INSTALLED_APPS = ['django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions', 'django.contrib.staticfiles', 'confirmation', 'zerver', 'social_django', 'django_scim', 'django_otp', 'django_otp.plugins.otp_static', 'django_otp.plugins.otp_totp', 'two_factor', 'two_factor.plugins.phonenumber']
if USING_PGROONGA:
    INSTALLED_APPS += ['pgroonga']
INSTALLED_APPS += EXTRA_INSTALLED_APPS
ZILENCER_ENABLED = 'zilencer' in INSTALLED_APPS
CORPORATE_ENABLED = 'corporate' in INSTALLED_APPS
if not TORNADO_PORTS:
    TORNADO_PORTS = get_tornado_ports(config_file)
TORNADO_PROCESSES = len(TORNADO_PORTS)
RUNNING_INSIDE_TORNADO = False
SILENCED_SYSTEM_CHECKS = ['auth.W004', 'models.E034']
DATABASES: Dict[str, Dict[str, Any]] = {'default': {'ENGINE': 'django.db.backends.postgresql', 'NAME': get_config('postgresql', 'database_name', 'zulip'), 'USER': get_config('postgresql', 'database_user', 'zulip'), 'PASSWORD': '', 'HOST': '', 'SCHEMA': 'zulip', 'CONN_MAX_AGE': 600, 'OPTIONS': {'connection_factory': TimeTrackingConnection, 'cursor_factory': TimeTrackingCursor}}}
if DEVELOPMENT:
    LOCAL_DATABASE_PASSWORD = get_secret('local_database_password')
    DATABASES['default'].update(PASSWORD=LOCAL_DATABASE_PASSWORD, HOST='localhost')
elif REMOTE_POSTGRES_HOST != '':
    DATABASES['default'].update(HOST=REMOTE_POSTGRES_HOST, PORT=REMOTE_POSTGRES_PORT)
    if get_secret('postgres_password') is not None:
        DATABASES['default'].update(PASSWORD=get_secret('postgres_password'))
    if REMOTE_POSTGRES_SSLMODE != '':
        DATABASES['default']['OPTIONS']['sslmode'] = REMOTE_POSTGRES_SSLMODE
    else:
        DATABASES['default']['OPTIONS']['sslmode'] = 'verify-full'
elif get_config('postgresql', 'database_user', 'zulip') != 'zulip' and get_secret('postgres_password') is not None:
    DATABASES['default'].update(PASSWORD=get_secret('postgres_password'), HOST='localhost')
POSTGRESQL_MISSING_DICTIONARIES = bool(get_config('postgresql', 'missing_dictionaries', None))
DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'
USING_RABBITMQ = True
RABBITMQ_PASSWORD = get_secret('rabbitmq_password')
SESSION_ENGINE = 'zerver.lib.safe_session_cached_db'
MEMCACHED_PASSWORD = get_secret('memcached_password')
CACHES: Dict[str, Dict[str, object]] = {'default': {'BACKEND': 'zerver.lib.singleton_bmemcached.SingletonBMemcached', 'LOCATION': MEMCACHED_LOCATION, 'OPTIONS': {'socket_timeout': 3600, 'username': MEMCACHED_USERNAME, 'password': MEMCACHED_PASSWORD, 'pickle_protocol': 4}}, 'database': {'BACKEND': 'django.core.cache.backends.db.DatabaseCache', 'LOCATION': 'third_party_api_results', 'TIMEOUT': None, 'OPTIONS': {'MAX_ENTRIES': 100000000, 'CULL_FREQUENCY': 10}}}
RATE_LIMITING_RULES = {**DEFAULT_RATE_LIMITING_RULES, **RATE_LIMITING_RULES}
RATE_LIMITING_DOMAINS_FOR_TORNADO = ['api_by_user', 'api_by_ip']
RATE_LIMITING_MIRROR_REALM_RULES = [(60, 50), (300, 120), (3600, 600)]
DEBUG_RATE_LIMITING = DEBUG
REDIS_PASSWORD = get_secret('redis_password')
if DEVELOPMENT:
    TOR_EXIT_NODE_FILE_PATH = os.path.join(DEPLOY_ROOT, 'var/tor-exit-nodes.json')
else:
    TOR_EXIT_NODE_FILE_PATH = '/var/lib/zulip/tor-exit-nodes.json'
if PRODUCTION:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    LANGUAGE_COOKIE_SECURE = True
    SESSION_COOKIE_NAME = '__Host-sessionid'
    CSRF_COOKIE_NAME = '__Host-csrftoken'
CSRF_COOKIE_HTTPONLY = True
CSRF_FAILURE_VIEW = 'zerver.middleware.csrf_failure'
LANGUAGE_COOKIE_SAMESITE: Final = 'Lax'
if DEVELOPMENT:
    PASSWORD_HASHERS = ('django.contrib.auth.hashers.SHA1PasswordHasher', 'django.contrib.auth.hashers.PBKDF2PasswordHasher')
    INITIAL_PASSWORD_SALT = get_secret('initial_password_salt')
else:
    PASSWORD_HASHERS = ('django.contrib.auth.hashers.Argon2PasswordHasher', 'django.contrib.auth.hashers.PBKDF2PasswordHasher')
ROOT_DOMAIN_URI = EXTERNAL_URI_SCHEME + EXTERNAL_HOST
S3_KEY = get_secret('s3_key')
S3_SECRET_KEY = get_secret('s3_secret_key')
ANDROID_GCM_API_KEY = get_secret('android_gcm_api_key')
DROPBOX_APP_KEY = get_secret('dropbox_app_key')
BIG_BLUE_BUTTON_SECRET = get_secret('big_blue_button_secret')
TWITTER_CONSUMER_KEY = get_secret('twitter_consumer_key')
TWITTER_CONSUMER_SECRET = get_secret('twitter_consumer_secret')
TWITTER_ACCESS_TOKEN_KEY = get_secret('twitter_access_token_key')
TWITTER_ACCESS_TOKEN_SECRET = get_secret('twitter_access_token_secret')
INTERNAL_BOTS = [{'var_name': 'NOTIFICATION_BOT', 'email_template': 'notification-bot@%s', 'name': 'Notification Bot'}, {'var_name': 'EMAIL_GATEWAY_BOT', 'email_template': 'emailgateway@%s', 'name': 'Email Gateway'}, {'var_name': 'NAGIOS_SEND_BOT', 'email_template': 'nagios-send-bot@%s', 'name': 'Nagios Send Bot'}, {'var_name': 'NAGIOS_RECEIVE_BOT', 'email_template': 'nagios-receive-bot@%s', 'name': 'Nagios Receive Bot'}, {'var_name': 'WELCOME_BOT', 'email_template': 'welcome-bot@%s', 'name': 'Welcome Bot'}]
REALM_INTERNAL_BOTS: List[Dict[str, str]] = []
DISABLED_REALM_INTERNAL_BOTS = [{'var_name': 'REMINDER_BOT', 'email_template': 'reminder-bot@%s', 'name': 'Reminder Bot'}]
if PRODUCTION:
    INTERNAL_BOTS += [{'var_name': 'NAGIOS_STAGING_SEND_BOT', 'email_template': 'nagios-staging-send-bot@%s', 'name': 'Nagios Staging Send Bot'}, {'var_name': 'NAGIOS_STAGING_RECEIVE_BOT', 'email_template': 'nagios-staging-receive-bot@%s', 'name': 'Nagios Staging Receive Bot'}]
INTERNAL_BOT_DOMAIN = 'zulip.com'
CAMO_KEY = get_secret('camo_key') if CAMO_URI != '' else None
if STATIC_URL is None:
    if PRODUCTION or IS_DEV_DROPLET or os.getenv('EXTERNAL_HOST') is not None:
        STATIC_URL = urljoin(ROOT_DOMAIN_URI, '/static/')
    else:
        STATIC_URL = 'http://localhost:9991/static/'
LOCAL_AVATARS_DIR = os.path.join(LOCAL_UPLOADS_DIR, 'avatars') if LOCAL_UPLOADS_DIR else None
LOCAL_FILES_DIR = os.path.join(LOCAL_UPLOADS_DIR, 'files') if LOCAL_UPLOADS_DIR else None
if not DEBUG:
    STATICFILES_STORAGE = 'zerver.lib.storage.ZulipStorage'
    if PRODUCTION:
        STATIC_ROOT = '/home/zulip/prod-static'
    else:
        STATIC_ROOT = os.path.abspath(os.path.join(DEPLOY_ROOT, 'prod-static/serve'))
LOCALE_PATHS = (os.path.join(DEPLOY_ROOT, 'locale'),)
FILE_UPLOAD_MAX_MEMORY_SIZE = 0
if DEVELOPMENT or 'ZULIP_COLLECTING_STATIC' in os.environ:
    STATICFILES_DIRS = [os.path.join(DEPLOY_ROOT, 'static')]
if DEBUG:
    WEBPACK_BUNDLES = '../webpack/'
    WEBPACK_STATS_FILE = os.path.join(DEPLOY_ROOT, 'var', 'webpack-stats-dev.json')
else:
    WEBPACK_BUNDLES = 'webpack-bundles/'
    WEBPACK_STATS_FILE = os.path.join(DEPLOY_ROOT, 'webpack-stats-production.json')
LOADERS: List[Union[str, Tuple[object, ...]]] = ['django.template.loaders.filesystem.Loader', 'django.template.loaders.app_directories.Loader']
if PRODUCTION:
    LOADERS = [('django.template.loaders.cached.Loader', LOADERS)]
base_template_engine_settings: Dict[str, Any] = {'BACKEND': 'django.template.backends.jinja2.Jinja2', 'OPTIONS': {'environment': 'zproject.jinja2.environment', 'extensions': ['jinja2.ext.i18n'], 'context_processors': ['zerver.context_processors.zulip_default_context', 'django.template.context_processors.i18n']}}
default_template_engine_settings = deepcopy(base_template_engine_settings)
default_template_engine_settings.update(NAME='Jinja2', DIRS=[os.path.join(DEPLOY_ROOT, 'templates'), os.path.join(DEPLOY_ROOT, 'zerver', 'webhooks'), os.path.join('static' if DEBUG else STATIC_ROOT, 'generated', 'bots')], APP_DIRS=True)
non_html_template_engine_settings = deepcopy(base_template_engine_settings)
non_html_template_engine_settings.update(NAME='Jinja2_plaintext', DIRS=[os.path.join(DEPLOY_ROOT, 'templates')], APP_DIRS=False)
non_html_template_engine_settings['OPTIONS'].update(autoescape=False, trim_blocks=True, lstrip_blocks=True)
two_factor_template_options = deepcopy(default_template_engine_settings['OPTIONS'])
del two_factor_template_options['environment']
del two_factor_template_options['extensions']
two_factor_template_options['loaders'] = ['zproject.template_loaders.TwoFactorLoader']
two_factor_template_engine_settings = {'NAME': 'Two_Factor', 'BACKEND': 'django.template.backends.django.DjangoTemplates', 'DIRS': [], 'APP_DIRS': False, 'OPTIONS': two_factor_template_options}
TEMPLATES = [default_template_engine_settings, non_html_template_engine_settings, two_factor_template_engine_settings]

def zulip_path(path: str) -> str:
    if False:
        return 10
    if DEVELOPMENT:
        if path.startswith('/var/log'):
            path = os.path.join(DEVELOPMENT_LOG_DIRECTORY, os.path.basename(path))
        else:
            path = os.path.join(os.path.join(DEPLOY_ROOT, 'var'), os.path.basename(path))
    return path
SERVER_LOG_PATH = zulip_path('/var/log/zulip/server.log')
ERROR_FILE_LOG_PATH = zulip_path('/var/log/zulip/errors.log')
MANAGEMENT_LOG_PATH = zulip_path('/var/log/zulip/manage.log')
WORKER_LOG_PATH = zulip_path('/var/log/zulip/workers.log')
SLOW_QUERIES_LOG_PATH = zulip_path('/var/log/zulip/slow_queries.log')
JSON_PERSISTENT_QUEUE_FILENAME_PATTERN = zulip_path('/home/zulip/tornado/event_queues%s.json')
EMAIL_LOG_PATH = zulip_path('/var/log/zulip/send_email.log')
EMAIL_MIRROR_LOG_PATH = zulip_path('/var/log/zulip/email_mirror.log')
EMAIL_DELIVERER_LOG_PATH = zulip_path('/var/log/zulip/email_deliverer.log')
EMAIL_CONTENT_LOG_PATH = zulip_path('/var/log/zulip/email_content.log')
LDAP_LOG_PATH = zulip_path('/var/log/zulip/ldap.log')
LDAP_SYNC_LOG_PATH = zulip_path('/var/log/zulip/sync_ldap_user_data.log')
QUEUE_ERROR_DIR = zulip_path('/var/log/zulip/queue_error')
QUEUE_STATS_DIR = zulip_path('/var/log/zulip/queue_stats')
DIGEST_LOG_PATH = zulip_path('/var/log/zulip/digest.log')
ANALYTICS_LOG_PATH = zulip_path('/var/log/zulip/analytics.log')
ANALYTICS_LOCK_DIR = zulip_path('/home/zulip/deployments/analytics-lock-dir')
WEBHOOK_LOG_PATH = zulip_path('/var/log/zulip/webhooks_errors.log')
WEBHOOK_ANOMALOUS_PAYLOADS_LOG_PATH = zulip_path('/var/log/zulip/webhooks_anomalous_payloads.log')
WEBHOOK_UNSUPPORTED_EVENTS_LOG_PATH = zulip_path('/var/log/zulip/webhooks_unsupported_events.log')
SOFT_DEACTIVATION_LOG_PATH = zulip_path('/var/log/zulip/soft_deactivation.log')
TRACEMALLOC_DUMP_DIR = zulip_path('/var/log/zulip/tracemalloc')
DELIVER_SCHEDULED_MESSAGES_LOG_PATH = zulip_path('/var/log/zulip/deliver_scheduled_messages.log')
RETENTION_LOG_PATH = zulip_path('/var/log/zulip/message_retention.log')
AUTH_LOG_PATH = zulip_path('/var/log/zulip/auth.log')
SCIM_LOG_PATH = zulip_path('/var/log/zulip/scim.log')
ZULIP_WORKER_TEST_FILE = zulip_path('/var/log/zulip/zulip-worker-test-file')
if IS_WORKER:
    FILE_LOG_PATH = WORKER_LOG_PATH
else:
    FILE_LOG_PATH = SERVER_LOG_PATH
DEFAULT_ZULIP_HANDLERS = [*(['mail_admins'] if ERROR_REPORTING else []), 'console', 'file', 'errors_file']

def skip_200_and_304(record: logging.LogRecord) -> bool:
    if False:
        i = 10
        return i + 15
    return getattr(record, 'status_code', None) not in [200, 304]

def skip_site_packages_logs(record: logging.LogRecord) -> bool:
    if False:
        while True:
            i = 10
    return 'site-packages' not in record.pathname
LOGGING: Dict[str, Any] = {'version': 1, 'disable_existing_loggers': False, 'formatters': {'default': {'()': 'zerver.lib.logging_util.ZulipFormatter'}, 'webhook_request_data': {'()': 'zerver.lib.logging_util.ZulipWebhookFormatter'}}, 'filters': {'ZulipLimiter': {'()': 'zerver.lib.logging_util.ZulipLimiter'}, 'EmailLimiter': {'()': 'zerver.lib.logging_util.EmailLimiter'}, 'require_debug_false': {'()': 'django.utils.log.RequireDebugFalse'}, 'require_debug_true': {'()': 'django.utils.log.RequireDebugTrue'}, 'nop': {'()': 'zerver.lib.logging_util.ReturnTrue'}, 'require_really_deployed': {'()': 'zerver.lib.logging_util.RequireReallyDeployed'}, 'skip_200_and_304': {'()': 'django.utils.log.CallbackFilter', 'callback': skip_200_and_304}, 'skip_site_packages_logs': {'()': 'django.utils.log.CallbackFilter', 'callback': skip_site_packages_logs}}, 'handlers': {'mail_admins': {'level': 'ERROR', 'class': 'django.utils.log.AdminEmailHandler', 'filters': ['ZulipLimiter', 'require_debug_false', 'require_really_deployed'] if not DEBUG_ERROR_REPORTING else []}, 'auth_file': {'level': 'DEBUG', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'default', 'filename': AUTH_LOG_PATH}, 'console': {'level': 'DEBUG', 'class': 'logging.StreamHandler', 'formatter': 'default'}, 'file': {'level': 'DEBUG', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'default', 'filename': FILE_LOG_PATH}, 'errors_file': {'level': 'WARNING', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'default', 'filename': ERROR_FILE_LOG_PATH}, 'ldap_file': {'level': 'DEBUG', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'default', 'filename': LDAP_LOG_PATH}, 'scim_file': {'level': 'DEBUG', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'default', 'filename': SCIM_LOG_PATH}, 'slow_queries_file': {'level': 'INFO', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'default', 'filename': SLOW_QUERIES_LOG_PATH}, 'webhook_file': {'level': 'DEBUG', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'webhook_request_data', 'filename': WEBHOOK_LOG_PATH}, 'webhook_unsupported_file': {'level': 'DEBUG', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'webhook_request_data', 'filename': WEBHOOK_UNSUPPORTED_EVENTS_LOG_PATH}, 'webhook_anomalous_file': {'level': 'DEBUG', 'class': 'logging.handlers.WatchedFileHandler', 'formatter': 'webhook_request_data', 'filename': WEBHOOK_ANOMALOUS_PAYLOADS_LOG_PATH}}, 'loggers': {'': {'level': 'INFO', 'handlers': DEFAULT_ZULIP_HANDLERS}, 'django': {}, 'django.request': {'level': 'ERROR'}, 'django.security.DisallowedHost': {'handlers': ['file'], 'propagate': False}, 'django.server': {'filters': ['skip_200_and_304'], 'handlers': ['console', 'file'], 'propagate': False}, 'django.utils.autoreload': {'level': 'WARNING'}, 'django.template': {'level': 'DEBUG', 'filters': ['require_debug_true', 'skip_site_packages_logs'], 'handlers': ['console'], 'propagate': False}, 'django_auth_ldap': {'level': 'DEBUG', 'handlers': ['console', 'ldap_file', 'errors_file'], 'propagate': False}, 'django_scim': {'level': 'DEBUG', 'handlers': ['scim_file', 'errors_file'], 'propagate': False}, 'pika': {'level': 'WARNING', 'handlers': ['console', 'file', 'errors_file'], 'propagate': False}, 'requests': {'level': 'WARNING'}, 'zerver.lib.digest': {'level': 'DEBUG'}, 'zerver.management.commands.deliver_scheduled_emails': {'level': 'DEBUG'}, 'zerver.management.commands.enqueue_digest_emails': {'level': 'DEBUG'}, 'zerver.management.commands.deliver_scheduled_messages': {'level': 'DEBUG'}, 'zulip.auth': {'level': 'DEBUG', 'handlers': [*DEFAULT_ZULIP_HANDLERS, 'auth_file'], 'propagate': False}, 'zulip.ldap': {'level': 'DEBUG', 'handlers': ['console', 'ldap_file', 'errors_file'], 'propagate': False}, 'zulip.management': {'handlers': ['file', 'errors_file'], 'propagate': False}, 'zulip.queue': {'level': 'WARNING'}, 'zulip.retention': {'handlers': ['file', 'errors_file'], 'propagate': False}, 'zulip.slow_queries': {'level': 'INFO', 'handlers': ['slow_queries_file'], 'propagate': False}, 'zulip.soft_deactivation': {'handlers': ['file', 'errors_file'], 'propagate': False}, 'zulip.zerver.webhooks': {'level': 'DEBUG', 'handlers': ['file', 'errors_file', 'webhook_file'], 'propagate': False}, 'zulip.zerver.webhooks.unsupported': {'level': 'DEBUG', 'handlers': ['webhook_unsupported_file'], 'propagate': False}, 'zulip.zerver.webhooks.anomalous': {'level': 'DEBUG', 'handlers': ['webhook_anomalous_file'], 'propagate': False}}}
if DEVELOPMENT:
    CONTRIBUTOR_DATA_FILE_PATH = os.path.join(DEPLOY_ROOT, 'var/github-contributors.json')
else:
    CONTRIBUTOR_DATA_FILE_PATH = '/var/lib/zulip/github-contributors.json'
LOGIN_REDIRECT_URL = '/'
EVENT_QUEUE_LONGPOLL_TIMEOUT_SECONDS = 90
USING_LDAP = 'zproject.backends.ZulipLDAPAuthBackend' in AUTHENTICATION_BACKENDS
ONLY_LDAP = AUTHENTICATION_BACKENDS == ('zproject.backends.ZulipLDAPAuthBackend',)
USING_APACHE_SSO = 'zproject.backends.ZulipRemoteUserBackend' in AUTHENTICATION_BACKENDS
ONLY_SSO = AUTHENTICATION_BACKENDS == ('zproject.backends.ZulipRemoteUserBackend',)
if CUSTOM_HOME_NOT_LOGGED_IN is not None:
    HOME_NOT_LOGGED_IN = CUSTOM_HOME_NOT_LOGGED_IN
elif ONLY_SSO:
    HOME_NOT_LOGGED_IN = '/accounts/login/sso/'
else:
    HOME_NOT_LOGGED_IN = '/login/'
AUTHENTICATION_BACKENDS += ('zproject.backends.ZulipDummyBackend',)
POPULATE_PROFILE_VIA_LDAP = bool(AUTH_LDAP_SERVER_URI)
if POPULATE_PROFILE_VIA_LDAP and (not USING_LDAP):
    AUTHENTICATION_BACKENDS += ('zproject.backends.ZulipLDAPUserPopulator',)
else:
    POPULATE_PROFILE_VIA_LDAP = USING_LDAP or POPULATE_PROFILE_VIA_LDAP
if POPULATE_PROFILE_VIA_LDAP:
    import ldap
    if AUTH_LDAP_BIND_DN and ldap.OPT_REFERRALS not in AUTH_LDAP_CONNECTION_OPTIONS:
        AUTH_LDAP_CONNECTION_OPTIONS[ldap.OPT_REFERRALS] = 0
if REGISTER_LINK_DISABLED is None:
    REGISTER_LINK_DISABLED = ONLY_LDAP
SOCIAL_AUTH_FIELDS_STORED_IN_SESSION = ['subdomain', 'is_signup', 'mobile_flow_otp', 'desktop_flow_otp', 'multiuse_object_key']
SOCIAL_AUTH_LOGIN_ERROR_URL = '/login/'
if SOCIAL_AUTH_SUBDOMAIN in ROOT_SUBDOMAIN_ALIASES:
    ROOT_SUBDOMAIN_ALIASES.remove(SOCIAL_AUTH_SUBDOMAIN)
SOCIAL_AUTH_APPLE_CLIENT = SOCIAL_AUTH_APPLE_SERVICES_ID
SOCIAL_AUTH_APPLE_AUDIENCE = [id for id in [SOCIAL_AUTH_APPLE_CLIENT, SOCIAL_AUTH_APPLE_APP_ID] if id is not None]
if PRODUCTION:
    SOCIAL_AUTH_APPLE_SECRET = get_from_file_if_exists('/etc/zulip/apple-auth-key.p8')
else:
    SOCIAL_AUTH_APPLE_SECRET = get_from_file_if_exists('zproject/dev_apple.key')
SOCIAL_AUTH_GITHUB_SECRET = get_secret('social_auth_github_secret')
SOCIAL_AUTH_GITLAB_SECRET = get_secret('social_auth_gitlab_secret')
SOCIAL_AUTH_AZUREAD_OAUTH2_SECRET = get_secret('social_auth_azuread_oauth2_secret')
SOCIAL_AUTH_GITHUB_SCOPE = ['user:email']
if SOCIAL_AUTH_GITHUB_ORG_NAME or SOCIAL_AUTH_GITHUB_TEAM_ID:
    SOCIAL_AUTH_GITHUB_SCOPE.append('read:org')
SOCIAL_AUTH_GITHUB_ORG_KEY = SOCIAL_AUTH_GITHUB_KEY
SOCIAL_AUTH_GITHUB_ORG_SECRET = SOCIAL_AUTH_GITHUB_SECRET
SOCIAL_AUTH_GITHUB_TEAM_KEY = SOCIAL_AUTH_GITHUB_KEY
SOCIAL_AUTH_GITHUB_TEAM_SECRET = SOCIAL_AUTH_GITHUB_SECRET
SOCIAL_AUTH_GOOGLE_SECRET = get_secret('social_auth_google_secret')
GOOGLE_OAUTH2_CLIENT_SECRET = get_secret('google_oauth2_client_secret')
SOCIAL_AUTH_GOOGLE_KEY = SOCIAL_AUTH_GOOGLE_KEY or GOOGLE_OAUTH2_CLIENT_ID
SOCIAL_AUTH_GOOGLE_SECRET = SOCIAL_AUTH_GOOGLE_SECRET or GOOGLE_OAUTH2_CLIENT_SECRET
if PRODUCTION:
    SOCIAL_AUTH_SAML_SP_PUBLIC_CERT = get_from_file_if_exists('/etc/zulip/saml/zulip-cert.crt')
    SOCIAL_AUTH_SAML_SP_PRIVATE_KEY = get_from_file_if_exists('/etc/zulip/saml/zulip-private-key.key')
    if SOCIAL_AUTH_SAML_SP_PUBLIC_CERT and SOCIAL_AUTH_SAML_SP_PRIVATE_KEY:
        if 'logoutRequestSigned' not in SOCIAL_AUTH_SAML_SECURITY_CONFIG:
            SOCIAL_AUTH_SAML_SECURITY_CONFIG['logoutRequestSigned'] = True
        if 'logoutResponseSigned' not in SOCIAL_AUTH_SAML_SECURITY_CONFIG:
            SOCIAL_AUTH_SAML_SECURITY_CONFIG['logoutResponseSigned'] = True
if 'signatureAlgorithm' not in SOCIAL_AUTH_SAML_SECURITY_CONFIG:
    default_signature_alg = 'http://www.w3.org/2001/04/xmldsig-more#rsa-sha256'
    SOCIAL_AUTH_SAML_SECURITY_CONFIG['signatureAlgorithm'] = default_signature_alg
for (idp_name, idp_dict) in SOCIAL_AUTH_SAML_ENABLED_IDPS.items():
    if DEVELOPMENT:
        idp_dict['entity_id'] = get_secret('saml_entity_id', '')
        idp_dict['url'] = get_secret('saml_url', '')
        idp_dict['x509cert_path'] = 'zproject/dev_saml.cert'
    if 'x509cert' in idp_dict:
        continue
    if 'x509cert_path' in idp_dict:
        path = idp_dict['x509cert_path']
    else:
        path = f'/etc/zulip/saml/idps/{idp_name}.crt'
    idp_dict['x509cert'] = get_from_file_if_exists(path)
SOCIAL_AUTH_PIPELINE = ['social_core.pipeline.social_auth.social_details', 'zproject.backends.social_auth_associate_user', 'zproject.backends.social_auth_finish']
DEFAULT_FROM_EMAIL = ZULIP_ADMINISTRATOR
if EMAIL_BACKEND is not None:
    pass
elif DEVELOPMENT:
    EMAIL_BACKEND = 'zproject.email_backends.EmailLogBackEnd'
elif not EMAIL_HOST:
    WARN_NO_EMAIL = True
    EMAIL_BACKEND = 'django.core.mail.backends.dummy.EmailBackend'
else:
    EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_TIMEOUT = 15
if DEVELOPMENT:
    EMAIL_HOST = get_secret('email_host', '')
    EMAIL_PORT = int(get_secret('email_port', '25'))
    EMAIL_HOST_USER = get_secret('email_host_user', '')
    EMAIL_USE_TLS = get_secret('email_use_tls', '') == 'true'
EMAIL_HOST_PASSWORD = get_secret('email_password')
EMAIL_GATEWAY_PASSWORD = get_secret('email_gateway_password')
AUTH_LDAP_BIND_PASSWORD = get_secret('auth_ldap_bind_password', '')
if PRODUCTION:
    DEFAULT_EXCEPTION_REPORTER_FILTER = 'zerver.filters.ZulipExceptionReporterFilter'
PROFILE_ALL_REQUESTS = False
CROSS_REALM_BOT_EMAILS = {'notification-bot@zulip.com', 'welcome-bot@zulip.com', 'emailgateway@zulip.com'}
TWO_FACTOR_PATCH_ADMIN = False
SENTRY_DSN = os.environ.get('SENTRY_DSN', SENTRY_DSN)
SCIM_SERVICE_PROVIDER = {'USER_ADAPTER': 'zerver.lib.scim.ZulipSCIMUser', 'USER_FILTER_PARSER': 'zerver.lib.scim_filter.ZulipUserFilterQuery', 'NETLOC': EXTERNAL_HOST, 'SCHEME': EXTERNAL_URI_SCHEME, 'GET_EXTRA_MODEL_FILTER_KWARGS_GETTER': 'zerver.lib.scim.get_extra_model_filter_kwargs_getter', 'BASE_LOCATION_GETTER': 'zerver.lib.scim.base_scim_location_getter', 'AUTH_CHECK_MIDDLEWARE': 'zerver.middleware.ZulipSCIMAuthCheckMiddleware', 'AUTHENTICATION_SCHEMES': [{'type': 'bearer', 'name': 'Bearer', 'description': 'Bearer token'}]}