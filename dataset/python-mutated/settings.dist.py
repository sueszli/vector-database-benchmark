import os
from datetime import timedelta
from celery.schedules import crontab
from dojo import __version__
import environ
from netaddr import IPNetwork, IPSet
import json
root = environ.Path(__file__) - 3
env = environ.Env(DD_SITE_URL=(str, 'http://localhost:8080'), DD_DEBUG=(bool, False), DD_TEMPLATE_DEBUG=(bool, False), DD_LOG_LEVEL=(str, ''), DD_DJANGO_METRICS_ENABLED=(bool, False), DD_LOGIN_REDIRECT_URL=(str, '/'), DD_LOGIN_URL=(str, '/login'), DD_DJANGO_ADMIN_ENABLED=(bool, True), DD_SESSION_COOKIE_HTTPONLY=(bool, True), DD_CSRF_COOKIE_HTTPONLY=(bool, True), DD_SECURE_SSL_REDIRECT=(bool, False), DD_SECURE_CROSS_ORIGIN_OPENER_POLICY=(str, 'same-origin'), DD_SECURE_HSTS_INCLUDE_SUBDOMAINS=(bool, False), DD_SECURE_HSTS_SECONDS=(int, 31536000), DD_SESSION_COOKIE_SECURE=(bool, False), DD_SESSION_EXPIRE_AT_BROWSER_CLOSE=(bool, False), DD_SESSION_COOKIE_AGE=(int, 1209600), DD_CSRF_COOKIE_SECURE=(bool, False), DD_CSRF_TRUSTED_ORIGINS=(list, []), DD_SECURE_CONTENT_TYPE_NOSNIFF=(bool, True), DD_CSRF_COOKIE_SAMESITE=(str, 'Lax'), DD_SESSION_COOKIE_SAMESITE=(str, 'Lax'), DD_TIME_ZONE=(str, 'UTC'), DD_LANG=(str, 'en-us'), DD_TEAM_NAME=(str, 'Security Team'), DD_ADMINS=(str, 'DefectDojo:dojo@localhost,Admin:admin@localhost'), DD_WHITENOISE=(bool, False), DD_TRACK_MIGRATIONS=(bool, True), DD_SECURE_PROXY_SSL_HEADER=(bool, False), DD_TEST_RUNNER=(str, 'django.test.runner.DiscoverRunner'), DD_URL_PREFIX=(str, ''), DD_ROOT=(str, root('dojo')), DD_LANGUAGE_CODE=(str, 'en-us'), DD_SITE_ID=(int, 1), DD_USE_I18N=(bool, True), DD_USE_L10N=(bool, True), DD_USE_TZ=(bool, True), DD_MEDIA_URL=(str, '/media/'), DD_MEDIA_ROOT=(str, root('media')), DD_STATIC_URL=(str, '/static/'), DD_STATIC_ROOT=(str, root('static')), DD_CELERY_BROKER_URL=(str, ''), DD_CELERY_BROKER_SCHEME=(str, 'sqla+sqlite'), DD_CELERY_BROKER_USER=(str, ''), DD_CELERY_BROKER_PASSWORD=(str, ''), DD_CELERY_BROKER_HOST=(str, ''), DD_CELERY_BROKER_PORT=(int, -1), DD_CELERY_BROKER_PATH=(str, '/dojo.celerydb.sqlite'), DD_CELERY_BROKER_PARAMS=(str, ''), DD_CELERY_BROKER_TRANSPORT_OPTIONS=(str, ''), DD_CELERY_TASK_IGNORE_RESULT=(bool, True), DD_CELERY_RESULT_BACKEND=(str, 'django-db'), DD_CELERY_RESULT_EXPIRES=(int, 86400), DD_CELERY_BEAT_SCHEDULE_FILENAME=(str, root('dojo.celery.beat.db')), DD_CELERY_TASK_SERIALIZER=(str, 'pickle'), DD_CELERY_PASS_MODEL_BY_ID=(str, True), DD_FOOTER_VERSION=(str, ''), DD_FORCE_LOWERCASE_TAGS=(bool, True), DD_MAX_TAG_LENGTH=(int, 25), DD_DATABASE_ENGINE=(str, 'django.db.backends.mysql'), DD_DATABASE_HOST=(str, 'mysql'), DD_DATABASE_NAME=(str, 'defectdojo'), DD_TEST_DATABASE_NAME=(str, 'test_defectdojo'), DD_DATABASE_PASSWORD=(str, 'defectdojo'), DD_DATABASE_PORT=(int, 3306), DD_DATABASE_USER=(str, 'defectdojo'), DD_SECRET_KEY=(str, ''), DD_CREDENTIAL_AES_256_KEY=(str, '.'), DD_DATA_UPLOAD_MAX_MEMORY_SIZE=(int, 8388608), DD_FORGOT_PASSWORD=(bool, True), DD_PASSWORD_RESET_TIMEOUT=(int, 259200), DD_FORGOT_USERNAME=(bool, True), DD_SOCIAL_AUTH_SHOW_LOGIN_FORM=(bool, True), DD_SOCIAL_AUTH_CREATE_USER=(bool, True), DD_SOCIAL_LOGIN_AUTO_REDIRECT=(bool, False), DD_SOCIAL_AUTH_TRAILING_SLASH=(bool, True), DD_SOCIAL_AUTH_AUTH0_OAUTH2_ENABLED=(bool, False), DD_SOCIAL_AUTH_AUTH0_KEY=(str, ''), DD_SOCIAL_AUTH_AUTH0_SECRET=(str, ''), DD_SOCIAL_AUTH_AUTH0_DOMAIN=(str, ''), DD_SOCIAL_AUTH_AUTH0_SCOPE=(list, ['openid', 'profile', 'email']), DD_SOCIAL_AUTH_GOOGLE_OAUTH2_ENABLED=(bool, False), DD_SOCIAL_AUTH_GOOGLE_OAUTH2_KEY=(str, ''), DD_SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET=(str, ''), DD_SOCIAL_AUTH_GOOGLE_OAUTH2_WHITELISTED_DOMAINS=(list, ['']), DD_SOCIAL_AUTH_GOOGLE_OAUTH2_WHITELISTED_EMAILS=(list, ['']), DD_SOCIAL_AUTH_OKTA_OAUTH2_ENABLED=(bool, False), DD_SOCIAL_AUTH_OKTA_OAUTH2_KEY=(str, ''), DD_SOCIAL_AUTH_OKTA_OAUTH2_SECRET=(str, ''), DD_SOCIAL_AUTH_OKTA_OAUTH2_API_URL=(str, 'https://{your-org-url}/oauth2'), DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_ENABLED=(bool, False), DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_KEY=(str, ''), DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_SECRET=(str, ''), DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_TENANT_ID=(str, ''), DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_RESOURCE=(str, 'https://graph.microsoft.com/'), DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_GET_GROUPS=(bool, False), DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_GROUPS_FILTER=(str, ''), DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_CLEANUP_GROUPS=(bool, True), DD_SOCIAL_AUTH_GITLAB_OAUTH2_ENABLED=(bool, False), DD_SOCIAL_AUTH_GITLAB_PROJECT_AUTO_IMPORT=(bool, False), DD_SOCIAL_AUTH_GITLAB_PROJECT_IMPORT_TAGS=(bool, False), DD_SOCIAL_AUTH_GITLAB_PROJECT_IMPORT_URL=(bool, False), DD_SOCIAL_AUTH_GITLAB_PROJECT_MIN_ACCESS_LEVEL=(int, 20), DD_SOCIAL_AUTH_GITLAB_KEY=(str, ''), DD_SOCIAL_AUTH_GITLAB_SECRET=(str, ''), DD_SOCIAL_AUTH_GITLAB_API_URL=(str, 'https://gitlab.com'), DD_SOCIAL_AUTH_GITLAB_SCOPE=(list, ['read_user', 'openid']), DD_SOCIAL_AUTH_KEYCLOAK_OAUTH2_ENABLED=(bool, False), DD_SOCIAL_AUTH_KEYCLOAK_KEY=(str, ''), DD_SOCIAL_AUTH_KEYCLOAK_SECRET=(str, ''), DD_SOCIAL_AUTH_KEYCLOAK_PUBLIC_KEY=(str, ''), DD_SOCIAL_AUTH_KEYCLOAK_AUTHORIZATION_URL=(str, ''), DD_SOCIAL_AUTH_KEYCLOAK_ACCESS_TOKEN_URL=(str, ''), DD_SOCIAL_AUTH_KEYCLOAK_LOGIN_BUTTON_TEXT=(str, 'Login with Keycloak'), DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_OAUTH2_ENABLED=(bool, False), DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_URL=(str, ''), DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_API_URL=(str, ''), DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_KEY=(str, ''), DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_SECRET=(str, ''), DD_SAML2_ENABLED=(bool, False), DD_SAML2_AUTHENTICATION_BACKENDS=(str, 'djangosaml2.backends.Saml2Backend'), DD_SAML2_FORCE_AUTH=(bool, True), DD_SAML2_LOGIN_BUTTON_TEXT=(str, 'Login with SAML'), DD_SAML2_LOGOUT_URL=(str, ''), DD_SAML2_METADATA_AUTO_CONF_URL=(str, ''), DD_SAML2_METADATA_LOCAL_FILE_PATH=(str, ''), DD_SAML2_ENTITY_ID=(str, ''), DD_SAML2_CREATE_USER=(bool, False), DD_SAML2_ATTRIBUTES_MAP=(dict, {'Email': 'email', 'UserName': 'username', 'Firstname': 'first_name', 'Lastname': 'last_name'}), DD_SAML2_ALLOW_UNKNOWN_ATTRIBUTE=(bool, False), DD_AUTH_REMOTEUSER_ENABLED=(bool, False), DD_AUTH_REMOTEUSER_USERNAME_HEADER=(str, 'REMOTE_USER'), DD_AUTH_REMOTEUSER_EMAIL_HEADER=(str, ''), DD_AUTH_REMOTEUSER_FIRSTNAME_HEADER=(str, ''), DD_AUTH_REMOTEUSER_LASTNAME_HEADER=(str, ''), DD_AUTH_REMOTEUSER_GROUPS_HEADER=(str, ''), DD_AUTH_REMOTEUSER_GROUPS_CLEANUP=(bool, True), DD_AUTH_REMOTEUSER_TRUSTED_PROXY=(list, ['127.0.0.0/32']), DD_AUTH_REMOTEUSER_LOGIN_ONLY=(bool, False), DD_DOCUMENTATION_URL=(str, 'https://documentation.defectdojo.com'), DD_DISABLE_FINDING_MERGE=(bool, False), DD_SLA_NOTIFY_ACTIVE=(bool, False), DD_SLA_NOTIFY_ACTIVE_VERIFIED_ONLY=(bool, False), DD_SLA_NOTIFY_WITH_JIRA_ONLY=(bool, False), DD_SLA_NOTIFY_PRE_BREACH=(int, 3), DD_SLA_NOTIFY_POST_BREACH=(int, 7), DD_SLA_BUSINESS_DAYS=(bool, False), DD_SEARCH_MAX_RESULTS=(int, 100), DD_SIMILAR_FINDINGS_MAX_RESULTS=(int, 25), DD_MAX_AUTOCOMPLETE_WORDS=(int, 20000), DD_JIRA_SSL_VERIFY=(bool, True), DD_JIRA_EXTRA_ISSUE_TYPES=(str, ''), DD_LOGGING_HANDLER=(str, 'console'), DD_DEFAULT_SWAGGER_UI=(bool, True), DD_ALERT_REFRESH=(bool, True), DD_DISABLE_ALERT_COUNTER=(bool, False), DD_MAX_ALERTS_PER_USER=(int, 999), DD_TAG_PREFETCHING=(bool, True), DD_QUALYS_WAS_WEAKNESS_IS_VULN=(bool, False), DD_PARSER_EXCLUDE=(str, 'AWS Scout2 Scan'), DD_DUPE_DELETE_MAX_PER_RUN=(int, 200), DD_EDITABLE_MITIGATED_DATA=(bool, False), DD_TRACK_IMPORT_HISTORY=(bool, True), DD_FEATURE_FINDING_GROUPS=(bool, True), DD_JIRA_TEMPLATE_ROOT=(str, 'dojo/templates/issue-trackers'), DD_TEMPLATE_DIR_PREFIX=(str, 'dojo/templates/'), DD_DUPLICATE_CLUSTER_CASCADE_DELETE=(str, False), DD_RATE_LIMITER_ENABLED=(bool, False), DD_RATE_LIMITER_RATE=(str, '5/m'), DD_RATE_LIMITER_BLOCK=(bool, False), DD_RATE_LIMITER_ACCOUNT_LOCKOUT=(bool, False), DD_SONARQUBE_API_PARSER_HOTSPOTS=(bool, True), DD_ASYNC_FINDING_IMPORT=(bool, False), DD_ASYNC_FINDING_IMPORT_CHUNK_SIZE=(int, 100), DD_ASYNC_OBJECT_DELETE=(bool, False), DD_ASYNC_OBEJECT_DELETE_CHUNK_SIZE=(int, 100), DD_DELETE_PREVIEW=(bool, True), DD_FILE_UPLOAD_TYPES=(list, ['.txt', '.pdf', '.json', '.xml', '.csv', '.yml', '.png', '.jpeg', '.sarif', '.xslx', '.doc', '.html', '.js', '.nessus', '.zip']), DD_SCAN_FILE_MAX_SIZE=(int, 100), DD_API_TOKENS_ENABLED=(bool, True), DD_ADDITIONAL_HEADERS=(dict, {}), DD_HASHCODE_FIELDS_PER_SCANNER=(str, ''), DD_DEDUPLICATION_ALGORITHM_PER_PARSER=(str, ''), DD_CREATE_CLOUD_BANNER=(bool, True))

def generate_url(scheme, double_slashes, user, password, host, port, path, params):
    if False:
        while True:
            i = 10
    result_list = []
    result_list.append(scheme)
    result_list.append(':')
    if double_slashes:
        result_list.append('//')
    result_list.append(user)
    if len(password) > 0:
        result_list.append(':')
        result_list.append(password)
    if len(user) > 0 or len(password) > 0:
        result_list.append('@')
    result_list.append(host)
    if port >= 0:
        result_list.append(':')
        result_list.append(str(port))
    if len(path) > 0 and path[0] != '/':
        result_list.append('/')
    result_list.append(path)
    if len(params) > 0 and params[0] != '?':
        result_list.append('?')
    result_list.append(params)
    return ''.join(result_list)
if os.path.isfile(root('dojo/settings/.env.prod')) or 'DD_ENV_PATH' in os.environ:
    env.read_env(root('dojo/settings/' + env.str('DD_ENV_PATH', '.env.prod')))
DEBUG = env('DD_DEBUG')
TEMPLATE_DEBUG = env('DD_TEMPLATE_DEBUG')
SITE_URL = env('DD_SITE_URL')
ALLOWED_HOSTS = tuple(env.list('DD_ALLOWED_HOSTS', default=['localhost', '127.0.0.1']))
SECRET_KEY = env('DD_SECRET_KEY')
TIME_ZONE = env('DD_TIME_ZONE')
LANGUAGE_CODE = env('DD_LANGUAGE_CODE')
SITE_ID = env('DD_SITE_ID')
USE_I18N = env('DD_USE_I18N')
USE_L10N = env('DD_USE_L10N')
USE_TZ = env('DD_USE_TZ')
TEST_RUNNER = env('DD_TEST_RUNNER')
ALERT_REFRESH = env('DD_ALERT_REFRESH')
DISABLE_ALERT_COUNTER = env('DD_DISABLE_ALERT_COUNTER')
MAX_ALERTS_PER_USER = env('DD_MAX_ALERTS_PER_USER')
TAG_PREFETCHING = env('DD_TAG_PREFETCHING')
if os.getenv('DD_DATABASE_URL') is not None:
    DATABASES = {'default': env.db('DD_DATABASE_URL')}
else:
    DATABASES = {'default': {'ENGINE': env('DD_DATABASE_ENGINE'), 'NAME': env('DD_DATABASE_NAME'), 'TEST': {'NAME': env('DD_TEST_DATABASE_NAME')}, 'USER': env('DD_DATABASE_USER'), 'PASSWORD': env('DD_DATABASE_PASSWORD'), 'HOST': env('DD_DATABASE_HOST'), 'PORT': env('DD_DATABASE_PORT')}}
if env('DD_TRACK_MIGRATIONS'):
    MIGRATION_MODULES = {'dojo': 'dojo.db_migrations'}
DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'
DOJO_ROOT = env('DD_ROOT')
MEDIA_ROOT = env('DD_MEDIA_ROOT')
MEDIA_URL = env('DD_MEDIA_URL')
STATIC_ROOT = env('DD_STATIC_ROOT')
STATIC_URL = env('DD_STATIC_URL')
STATICFILES_DIRS = (os.path.join(os.path.dirname(DOJO_ROOT), 'components', 'node_modules'),)
STATICFILES_FINDERS = ('django.contrib.staticfiles.finders.FileSystemFinder', 'django.contrib.staticfiles.finders.AppDirectoriesFinder')
FILE_UPLOAD_HANDLERS = ('django.core.files.uploadhandler.TemporaryFileUploadHandler',)
DATA_UPLOAD_MAX_MEMORY_SIZE = env('DD_DATA_UPLOAD_MAX_MEMORY_SIZE')
ROOT_URLCONF = 'dojo.urls'
WSGI_APPLICATION = 'dojo.wsgi.application'
URL_PREFIX = env('DD_URL_PREFIX')
LOGIN_REDIRECT_URL = env('DD_LOGIN_REDIRECT_URL')
LOGIN_URL = env('DD_LOGIN_URL')
AUTHENTICATION_BACKENDS = ('social_core.backends.auth0.Auth0OAuth2', 'social_core.backends.google.GoogleOAuth2', 'dojo.okta.OktaOAuth2', 'social_core.backends.azuread_tenant.AzureADTenantOAuth2', 'social_core.backends.gitlab.GitLabOAuth2', 'social_core.backends.keycloak.KeycloakOAuth2', 'social_core.backends.github_enterprise.GithubEnterpriseOAuth2', 'dojo.remote_user.RemoteUserBackend', 'django.contrib.auth.backends.RemoteUserBackend', 'django.contrib.auth.backends.ModelBackend')
PASSWORD_HASHERS = ['django.contrib.auth.hashers.Argon2PasswordHasher', 'django.contrib.auth.hashers.PBKDF2PasswordHasher', 'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher', 'django.contrib.auth.hashers.BCryptSHA256PasswordHasher', 'django.contrib.auth.hashers.BCryptPasswordHasher', 'django.contrib.auth.hashers.SHA1PasswordHasher', 'django.contrib.auth.hashers.MD5PasswordHasher', 'django.contrib.auth.hashers.UnsaltedSHA1PasswordHasher', 'django.contrib.auth.hashers.UnsaltedMD5PasswordHasher', 'django.contrib.auth.hashers.CryptPasswordHasher']
SOCIAL_AUTH_PIPELINE = ('social_core.pipeline.social_auth.social_details', 'dojo.pipeline.social_uid', 'social_core.pipeline.social_auth.auth_allowed', 'social_core.pipeline.social_auth.social_user', 'social_core.pipeline.user.get_username', 'social_core.pipeline.social_auth.associate_by_email', 'dojo.pipeline.create_user', 'dojo.pipeline.modify_permissions', 'social_core.pipeline.social_auth.associate_user', 'social_core.pipeline.social_auth.load_extra_data', 'social_core.pipeline.user.user_details', 'dojo.pipeline.update_azure_groups', 'dojo.pipeline.update_product_access')
CLASSIC_AUTH_ENABLED = True
FORGOT_PASSWORD = env('DD_FORGOT_PASSWORD')
FORGOT_USERNAME = env('DD_FORGOT_USERNAME')
PASSWORD_RESET_TIMEOUT = env('DD_PASSWORD_RESET_TIMEOUT')
SHOW_LOGIN_FORM = env('DD_SOCIAL_AUTH_SHOW_LOGIN_FORM')
SOCIAL_LOGIN_AUTO_REDIRECT = env('DD_SOCIAL_LOGIN_AUTO_REDIRECT')
SOCIAL_AUTH_CREATE_USER = env('DD_SOCIAL_AUTH_CREATE_USER')
SOCIAL_AUTH_STRATEGY = 'social_django.strategy.DjangoStrategy'
SOCIAL_AUTH_STORAGE = 'social_django.models.DjangoStorage'
SOCIAL_AUTH_ADMIN_USER_SEARCH_FIELDS = ['username', 'first_name', 'last_name', 'email']
SOCIAL_AUTH_USERNAME_IS_FULL_EMAIL = True
GOOGLE_OAUTH_ENABLED = env('DD_SOCIAL_AUTH_GOOGLE_OAUTH2_ENABLED')
SOCIAL_AUTH_GOOGLE_OAUTH2_KEY = env('DD_SOCIAL_AUTH_GOOGLE_OAUTH2_KEY')
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = env('DD_SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET')
SOCIAL_AUTH_GOOGLE_OAUTH2_WHITELISTED_DOMAINS = env('DD_SOCIAL_AUTH_GOOGLE_OAUTH2_WHITELISTED_DOMAINS')
SOCIAL_AUTH_GOOGLE_OAUTH2_WHITELISTED_EMAILS = env('DD_SOCIAL_AUTH_GOOGLE_OAUTH2_WHITELISTED_EMAILS')
SOCIAL_AUTH_LOGIN_ERROR_URL = '/login'
SOCIAL_AUTH_BACKEND_ERROR_URL = '/login'
OKTA_OAUTH_ENABLED = env('DD_SOCIAL_AUTH_OKTA_OAUTH2_ENABLED')
SOCIAL_AUTH_OKTA_OAUTH2_KEY = env('DD_SOCIAL_AUTH_OKTA_OAUTH2_KEY')
SOCIAL_AUTH_OKTA_OAUTH2_SECRET = env('DD_SOCIAL_AUTH_OKTA_OAUTH2_SECRET')
SOCIAL_AUTH_OKTA_OAUTH2_API_URL = env('DD_SOCIAL_AUTH_OKTA_OAUTH2_API_URL')
AZUREAD_TENANT_OAUTH2_ENABLED = env('DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_ENABLED')
SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_KEY = env('DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_KEY')
SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_SECRET = env('DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_SECRET')
SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_TENANT_ID = env('DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_TENANT_ID')
SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_RESOURCE = env('DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_RESOURCE')
AZUREAD_TENANT_OAUTH2_GET_GROUPS = env('DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_GET_GROUPS')
AZUREAD_TENANT_OAUTH2_GROUPS_FILTER = env('DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_GROUPS_FILTER')
AZUREAD_TENANT_OAUTH2_CLEANUP_GROUPS = env('DD_SOCIAL_AUTH_AZUREAD_TENANT_OAUTH2_CLEANUP_GROUPS')
GITLAB_OAUTH2_ENABLED = env('DD_SOCIAL_AUTH_GITLAB_OAUTH2_ENABLED')
GITLAB_PROJECT_AUTO_IMPORT = env('DD_SOCIAL_AUTH_GITLAB_PROJECT_AUTO_IMPORT')
GITLAB_PROJECT_IMPORT_TAGS = env('DD_SOCIAL_AUTH_GITLAB_PROJECT_IMPORT_TAGS')
GITLAB_PROJECT_IMPORT_URL = env('DD_SOCIAL_AUTH_GITLAB_PROJECT_IMPORT_URL')
GITLAB_PROJECT_MIN_ACCESS_LEVEL = env('DD_SOCIAL_AUTH_GITLAB_PROJECT_MIN_ACCESS_LEVEL')
SOCIAL_AUTH_GITLAB_KEY = env('DD_SOCIAL_AUTH_GITLAB_KEY')
SOCIAL_AUTH_GITLAB_SECRET = env('DD_SOCIAL_AUTH_GITLAB_SECRET')
SOCIAL_AUTH_GITLAB_API_URL = env('DD_SOCIAL_AUTH_GITLAB_API_URL')
SOCIAL_AUTH_GITLAB_SCOPE = env('DD_SOCIAL_AUTH_GITLAB_SCOPE')
if GITLAB_PROJECT_AUTO_IMPORT:
    SOCIAL_AUTH_GITLAB_SCOPE += ['read_repository']
AUTH0_OAUTH2_ENABLED = env('DD_SOCIAL_AUTH_AUTH0_OAUTH2_ENABLED')
SOCIAL_AUTH_AUTH0_KEY = env('DD_SOCIAL_AUTH_AUTH0_KEY')
SOCIAL_AUTH_AUTH0_SECRET = env('DD_SOCIAL_AUTH_AUTH0_SECRET')
SOCIAL_AUTH_AUTH0_DOMAIN = env('DD_SOCIAL_AUTH_AUTH0_DOMAIN')
SOCIAL_AUTH_AUTH0_SCOPE = env('DD_SOCIAL_AUTH_AUTH0_SCOPE')
SOCIAL_AUTH_TRAILING_SLASH = env('DD_SOCIAL_AUTH_TRAILING_SLASH')
KEYCLOAK_OAUTH2_ENABLED = env('DD_SOCIAL_AUTH_KEYCLOAK_OAUTH2_ENABLED')
SOCIAL_AUTH_KEYCLOAK_KEY = env('DD_SOCIAL_AUTH_KEYCLOAK_KEY')
SOCIAL_AUTH_KEYCLOAK_SECRET = env('DD_SOCIAL_AUTH_KEYCLOAK_SECRET')
SOCIAL_AUTH_KEYCLOAK_PUBLIC_KEY = env('DD_SOCIAL_AUTH_KEYCLOAK_PUBLIC_KEY')
SOCIAL_AUTH_KEYCLOAK_AUTHORIZATION_URL = env('DD_SOCIAL_AUTH_KEYCLOAK_AUTHORIZATION_URL')
SOCIAL_AUTH_KEYCLOAK_ACCESS_TOKEN_URL = env('DD_SOCIAL_AUTH_KEYCLOAK_ACCESS_TOKEN_URL')
SOCIAL_AUTH_KEYCLOAK_LOGIN_BUTTON_TEXT = env('DD_SOCIAL_AUTH_KEYCLOAK_LOGIN_BUTTON_TEXT')
GITHUB_ENTERPRISE_OAUTH2_ENABLED = env('DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_OAUTH2_ENABLED')
SOCIAL_AUTH_GITHUB_ENTERPRISE_URL = env('DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_URL')
SOCIAL_AUTH_GITHUB_ENTERPRISE_API_URL = env('DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_API_URL')
SOCIAL_AUTH_GITHUB_ENTERPRISE_KEY = env('DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_KEY')
SOCIAL_AUTH_GITHUB_ENTERPRISE_SECRET = env('DD_SOCIAL_AUTH_GITHUB_ENTERPRISE_SECRET')
DOCUMENTATION_URL = env('DD_DOCUMENTATION_URL')
SLA_NOTIFY_ACTIVE = env('DD_SLA_NOTIFY_ACTIVE')
SLA_NOTIFY_ACTIVE_VERIFIED_ONLY = env('DD_SLA_NOTIFY_ACTIVE_VERIFIED_ONLY')
SLA_NOTIFY_WITH_JIRA_ONLY = env('DD_SLA_NOTIFY_WITH_JIRA_ONLY')
SLA_NOTIFY_PRE_BREACH = env('DD_SLA_NOTIFY_PRE_BREACH')
SLA_NOTIFY_POST_BREACH = env('DD_SLA_NOTIFY_POST_BREACH')
SLA_BUSINESS_DAYS = env('DD_SLA_BUSINESS_DAYS')
SEARCH_MAX_RESULTS = env('DD_SEARCH_MAX_RESULTS')
SIMILAR_FINDINGS_MAX_RESULTS = env('DD_SIMILAR_FINDINGS_MAX_RESULTS')
MAX_AUTOCOMPLETE_WORDS = env('DD_MAX_AUTOCOMPLETE_WORDS')
LOGIN_EXEMPT_URLS = ('^%sstatic/' % URL_PREFIX, '^%swebhook/([\\w-]+)$' % URL_PREFIX, '^%swebhook/' % URL_PREFIX, '^%sjira/webhook/([\\w-]+)$' % URL_PREFIX, '^%sjira/webhook/' % URL_PREFIX, '^%sreports/cover$' % URL_PREFIX, '^%sfinding/image/(?P<token>[^/]+)$' % URL_PREFIX, '^%sapi/v2/' % URL_PREFIX, 'complete/', 'empty_questionnaire/([\\d]+)/answer', '^%spassword_reset/' % URL_PREFIX, '^%sforgot_username' % URL_PREFIX, '^%sreset/' % URL_PREFIX)
AUTH_PASSWORD_VALIDATORS = [{'NAME': 'dojo.user.validators.DojoCommonPasswordValidator'}, {'NAME': 'dojo.user.validators.MinLengthValidator'}, {'NAME': 'dojo.user.validators.MaxLengthValidator'}, {'NAME': 'dojo.user.validators.NumberValidator'}, {'NAME': 'dojo.user.validators.UppercaseValidator'}, {'NAME': 'dojo.user.validators.LowercaseValidator'}, {'NAME': 'dojo.user.validators.SymbolValidator'}]
RATE_LIMITER_ENABLED = env('DD_RATE_LIMITER_ENABLED')
RATE_LIMITER_RATE = env('DD_RATE_LIMITER_RATE')
RATE_LIMITER_BLOCK = env('DD_RATE_LIMITER_BLOCK')
RATE_LIMITER_ACCOUNT_LOCKOUT = env('DD_RATE_LIMITER_ACCOUNT_LOCKOUT')
SECURE_SSL_REDIRECT = env('DD_SECURE_SSL_REDIRECT')
SECURE_CONTENT_TYPE_NOSNIFF = env('DD_SECURE_CONTENT_TYPE_NOSNIFF')
SESSION_COOKIE_HTTPONLY = env('DD_SESSION_COOKIE_HTTPONLY')
CSRF_COOKIE_HTTPONLY = env('DD_CSRF_COOKIE_HTTPONLY')
SESSION_COOKIE_SECURE = env('DD_SESSION_COOKIE_SECURE')
SESSION_COOKIE_SAMESITE = env('DD_SESSION_COOKIE_SAMESITE')
CSRF_COOKIE_SECURE = env('DD_CSRF_COOKIE_SECURE')
CSRF_COOKIE_SAMESITE = env('DD_CSRF_COOKIE_SAMESITE')
if env('DD_CSRF_TRUSTED_ORIGINS') != ['[]']:
    CSRF_TRUSTED_ORIGINS = env('DD_CSRF_TRUSTED_ORIGINS')
SECURE_CROSS_ORIGIN_OPENER_POLICY = env('DD_SECURE_CROSS_ORIGIN_OPENER_POLICY') if env('DD_SECURE_CROSS_ORIGIN_OPENER_POLICY') != 'None' else None
if env('DD_SECURE_PROXY_SSL_HEADER'):
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
if env('DD_SECURE_HSTS_INCLUDE_SUBDOMAINS'):
    SECURE_HSTS_SECONDS = env('DD_SECURE_HSTS_SECONDS')
    SECURE_HSTS_INCLUDE_SUBDOMAINS = env('DD_SECURE_HSTS_INCLUDE_SUBDOMAINS')
SESSION_EXPIRE_AT_BROWSER_CLOSE = env('DD_SESSION_EXPIRE_AT_BROWSER_CLOSE')
SESSION_COOKIE_AGE = env('DD_SESSION_COOKIE_AGE')
CREDENTIAL_AES_256_KEY = env('DD_CREDENTIAL_AES_256_KEY')
DB_KEY = env('DD_CREDENTIAL_AES_256_KEY')
TEAM_NAME = env('DD_TEAM_NAME')
FOOTER_VERSION = env('DD_FOOTER_VERSION')
FORCE_LOWERCASE_TAGS = env('DD_FORCE_LOWERCASE_TAGS')
MAX_TAG_LENGTH = env('DD_MAX_TAG_LENGTH')
from email.utils import getaddresses
ADMINS = getaddresses([env('DD_ADMINS')])
MANAGERS = ADMINS
DJANGO_ADMIN_ENABLED = env('DD_DJANGO_ADMIN_ENABLED')
API_TOKENS_ENABLED = env('DD_API_TOKENS_ENABLED')
REST_FRAMEWORK = {'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema', 'DEFAULT_AUTHENTICATION_CLASSES': ('rest_framework.authentication.SessionAuthentication', 'rest_framework.authentication.BasicAuthentication'), 'DEFAULT_PERMISSION_CLASSES': ('rest_framework.permissions.DjangoModelPermissions',), 'DEFAULT_RENDERER_CLASSES': ('rest_framework.renderers.JSONRenderer',), 'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.LimitOffsetPagination', 'PAGE_SIZE': 25, 'EXCEPTION_HANDLER': 'dojo.api_v2.exception_handler.custom_exception_handler'}
if API_TOKENS_ENABLED:
    REST_FRAMEWORK['DEFAULT_AUTHENTICATION_CLASSES'] += ('rest_framework.authentication.TokenAuthentication',)
SWAGGER_SETTINGS = {'SECURITY_DEFINITIONS': {'basicAuth': {'type': 'basic'}, 'cookieAuth': {'type': 'apiKey', 'in': 'cookie', 'name': 'sessionid'}}, 'DOC_EXPANSION': 'none', 'JSON_EDITOR': True, 'SHOW_REQUEST_HEADERS': True}
if API_TOKENS_ENABLED:
    SWAGGER_SETTINGS['SECURITY_DEFINITIONS']['tokenAuth'] = {'type': 'apiKey', 'in': 'header', 'name': 'Authorization'}
SPECTACULAR_SETTINGS = {'TITLE': 'Defect Dojo API v2', 'DESCRIPTION': 'Defect Dojo - Open Source vulnerability Management made easy. Prefetch related parameters/responses not yet in the schema.', 'VERSION': __version__, 'SCHEMA_PATH_PREFIX': '/api/v2', 'PREPROCESSING_HOOKS': ['dojo.urls.drf_spectacular_preprocessing_filter_spec'], 'POSTPROCESSING_HOOKS': ['dojo.api_v2.prefetch.schema.prefetch_postprocessing_hook'], 'COMPONENT_SPLIT_REQUEST': True, 'SWAGGER_UI_SETTINGS': {'docExpansion': 'none'}}
if not env('DD_DEFAULT_SWAGGER_UI'):
    SPECTACULAR_SETTINGS['SWAGGER_UI_DIST'] = f'{STATIC_URL}drf-yasg/swagger-ui-dist'
    SPECTACULAR_SETTINGS['SWAGGER_UI_FAVICON_HREF'] = f'{STATIC_URL}drf-yasg/swagger-ui-dist/favicon-32x32.png'
TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates', 'APP_DIRS': True, 'OPTIONS': {'debug': env('DD_DEBUG'), 'context_processors': ['django.template.context_processors.debug', 'django.template.context_processors.request', 'django.contrib.auth.context_processors.auth', 'django.contrib.messages.context_processors.messages', 'social_django.context_processors.backends', 'social_django.context_processors.login_redirect', 'dojo.context_processors.globalize_vars', 'dojo.context_processors.bind_system_settings', 'dojo.context_processors.bind_alert_count', 'dojo.context_processors.bind_announcement']}}]
INSTALLED_APPS = ('django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions', 'django.contrib.sites', 'django.contrib.messages', 'django.contrib.staticfiles', 'polymorphic', 'django.contrib.admin', 'django.contrib.humanize', 'gunicorn', 'auditlog', 'dojo', 'watson', 'tagging', 'imagekit', 'multiselectfield', 'rest_framework', 'rest_framework.authtoken', 'dbbackup', 'django_celery_results', 'social_django', 'drf_yasg', 'drf_spectacular', 'tagulous', 'fontawesomefree')
DJANGO_MIDDLEWARE_CLASSES = ['django.middleware.common.CommonMiddleware', 'dojo.middleware.APITrailingSlashMiddleware', 'dojo.middleware.DojoSytemSettingsMiddleware', 'django.contrib.sessions.middleware.SessionMiddleware', 'django.middleware.csrf.CsrfViewMiddleware', 'django.middleware.security.SecurityMiddleware', 'django.contrib.auth.middleware.AuthenticationMiddleware', 'django.contrib.messages.middleware.MessageMiddleware', 'django.middleware.clickjacking.XFrameOptionsMiddleware', 'dojo.middleware.LoginRequiredMiddleware', 'dojo.middleware.AdditionalHeaderMiddleware', 'social_django.middleware.SocialAuthExceptionMiddleware', 'watson.middleware.SearchContextMiddleware', 'auditlog.middleware.AuditlogMiddleware', 'crum.CurrentRequestUserMiddleware', 'dojo.request_cache.middleware.RequestCacheMiddleware']
MIDDLEWARE = DJANGO_MIDDLEWARE_CLASSES
if env('DD_WHITENOISE'):
    WHITE_NOISE = ['whitenoise.middleware.WhiteNoiseMiddleware']
    MIDDLEWARE = MIDDLEWARE + WHITE_NOISE
EMAIL_CONFIG = env.email_url('DD_EMAIL_URL', default='smtp://user@:password@localhost:25')
vars().update(EMAIL_CONFIG)

def saml2_attrib_map_format(dict):
    if False:
        print('Hello World!')
    dout = {}
    for i in dict:
        dout[i] = (dict[i],)
    return dout
SAML2_ENABLED = env('DD_SAML2_ENABLED')
SAML2_LOGIN_BUTTON_TEXT = env('DD_SAML2_LOGIN_BUTTON_TEXT')
SAML2_LOGOUT_URL = env('DD_SAML2_LOGOUT_URL')
if SAML2_ENABLED:
    import saml2
    import saml2.saml
    from os import path
    SAML_METADATA = {}
    if len(env('DD_SAML2_METADATA_AUTO_CONF_URL')) > 0:
        SAML_METADATA['remote'] = [{'url': env('DD_SAML2_METADATA_AUTO_CONF_URL')}]
    if len(env('DD_SAML2_METADATA_LOCAL_FILE_PATH')) > 0:
        SAML_METADATA['local'] = [env('DD_SAML2_METADATA_LOCAL_FILE_PATH')]
    INSTALLED_APPS += ('djangosaml2',)
    MIDDLEWARE.append('djangosaml2.middleware.SamlSessionMiddleware')
    AUTHENTICATION_BACKENDS += (env('DD_SAML2_AUTHENTICATION_BACKENDS'),)
    LOGIN_EXEMPT_URLS += ('^%ssaml2/' % URL_PREFIX,)
    SAML_LOGOUT_REQUEST_PREFERRED_BINDING = saml2.BINDING_HTTP_POST
    SAML_IGNORE_LOGOUT_ERRORS = True
    SAML_DJANGO_USER_MAIN_ATTRIBUTE = 'username'
    SAML_USE_NAME_ID_AS_USERNAME = True
    SAML_CREATE_UNKNOWN_USER = env('DD_SAML2_CREATE_USER')
    SAML_ATTRIBUTE_MAPPING = saml2_attrib_map_format(env('DD_SAML2_ATTRIBUTES_MAP'))
    SAML_FORCE_AUTH = env('DD_SAML2_FORCE_AUTH')
    SAML_ALLOW_UNKNOWN_ATTRIBUTES = env('DD_SAML2_ALLOW_UNKNOWN_ATTRIBUTE')
    BASEDIR = path.dirname(path.abspath(__file__))
    if len(env('DD_SAML2_ENTITY_ID')) == 0:
        SAML2_ENTITY_ID = '%s/saml2/metadata/' % SITE_URL
    else:
        SAML2_ENTITY_ID = env('DD_SAML2_ENTITY_ID')
    SAML_CONFIG = {'xmlsec_binary': '/usr/bin/xmlsec1', 'entityid': '%s' % SAML2_ENTITY_ID, 'attribute_map_dir': path.join(BASEDIR, 'attribute-maps'), 'allow_unknown_attributes': SAML_ALLOW_UNKNOWN_ATTRIBUTES, 'service': {'sp': {'name': 'Defect_Dojo', 'name_id_format': saml2.saml.NAMEID_FORMAT_TRANSIENT, 'want_response_signed': False, 'want_assertions_signed': True, 'force_authn': SAML_FORCE_AUTH, 'allow_unsolicited': True, 'endpoints': {'assertion_consumer_service': [('%s/saml2/acs/' % SITE_URL, saml2.BINDING_HTTP_POST)], 'single_logout_service': [('%s/saml2/ls/' % SITE_URL, saml2.BINDING_HTTP_REDIRECT), ('%s/saml2/ls/post' % SITE_URL, saml2.BINDING_HTTP_POST)]}, 'required_attributes': ['Email', 'UserName'], 'optional_attributes': ['Firstname', 'Lastname']}}, 'metadata': SAML_METADATA, 'debug': 0, 'contact_person': [{'given_name': 'Lorenzo', 'sur_name': 'Gil', 'company': 'Yaco Sistemas', 'email_address': 'lgs@yaco.es', 'contact_type': 'technical'}, {'given_name': 'Angel', 'sur_name': 'Fernandez', 'company': 'Yaco Sistemas', 'email_address': 'angel@yaco.es', 'contact_type': 'administrative'}], 'organization': {'name': [('Yaco Sistemas', 'es'), ('Yaco Systems', 'en')], 'display_name': [('Yaco', 'es'), ('Yaco', 'en')], 'url': [('http://www.yaco.es', 'es'), ('http://www.yaco.com', 'en')]}, 'valid_for': 24}
AUTH_REMOTEUSER_ENABLED = env('DD_AUTH_REMOTEUSER_ENABLED')
if AUTH_REMOTEUSER_ENABLED:
    AUTH_REMOTEUSER_USERNAME_HEADER = env('DD_AUTH_REMOTEUSER_USERNAME_HEADER')
    AUTH_REMOTEUSER_EMAIL_HEADER = env('DD_AUTH_REMOTEUSER_EMAIL_HEADER')
    AUTH_REMOTEUSER_FIRSTNAME_HEADER = env('DD_AUTH_REMOTEUSER_FIRSTNAME_HEADER')
    AUTH_REMOTEUSER_LASTNAME_HEADER = env('DD_AUTH_REMOTEUSER_LASTNAME_HEADER')
    AUTH_REMOTEUSER_GROUPS_HEADER = env('DD_AUTH_REMOTEUSER_GROUPS_HEADER')
    AUTH_REMOTEUSER_GROUPS_CLEANUP = env('DD_AUTH_REMOTEUSER_GROUPS_CLEANUP')
    AUTH_REMOTEUSER_TRUSTED_PROXY = IPSet()
    for ip_range in env('DD_AUTH_REMOTEUSER_TRUSTED_PROXY'):
        AUTH_REMOTEUSER_TRUSTED_PROXY.add(IPNetwork(ip_range))
    if env('DD_AUTH_REMOTEUSER_LOGIN_ONLY'):
        RemoteUserMiddleware = 'dojo.remote_user.PersistentRemoteUserMiddleware'
    else:
        RemoteUserMiddleware = 'dojo.remote_user.RemoteUserMiddleware'
    for i in range(len(MIDDLEWARE)):
        if MIDDLEWARE[i] == 'django.contrib.auth.middleware.AuthenticationMiddleware':
            MIDDLEWARE.insert(i + 1, RemoteUserMiddleware)
            break
    REST_FRAMEWORK['DEFAULT_AUTHENTICATION_CLASSES'] = ('dojo.remote_user.RemoteUserAuthentication',) + REST_FRAMEWORK['DEFAULT_AUTHENTICATION_CLASSES']
    SWAGGER_SETTINGS['SECURITY_DEFINITIONS']['remoteUserAuth'] = {'type': 'apiKey', 'in': 'header', 'name': AUTH_REMOTEUSER_USERNAME_HEADER[5:].replace('_', '-')}
CELERY_BROKER_URL = env('DD_CELERY_BROKER_URL') if len(env('DD_CELERY_BROKER_URL')) > 0 else generate_url(env('DD_CELERY_BROKER_SCHEME'), True, env('DD_CELERY_BROKER_USER'), env('DD_CELERY_BROKER_PASSWORD'), env('DD_CELERY_BROKER_HOST'), env('DD_CELERY_BROKER_PORT'), env('DD_CELERY_BROKER_PATH'), env('DD_CELERY_BROKER_PARAMS'))
CELERY_TASK_IGNORE_RESULT = env('DD_CELERY_TASK_IGNORE_RESULT')
CELERY_RESULT_BACKEND = env('DD_CELERY_RESULT_BACKEND')
CELERY_TIMEZONE = TIME_ZONE
CELERY_RESULT_EXPIRES = env('DD_CELERY_RESULT_EXPIRES')
CELERY_BEAT_SCHEDULE_FILENAME = env('DD_CELERY_BEAT_SCHEDULE_FILENAME')
CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml']
CELERY_TASK_SERIALIZER = env('DD_CELERY_TASK_SERIALIZER')
CELERY_PASS_MODEL_BY_ID = env('DD_CELERY_PASS_MODEL_BY_ID')
if len(env('DD_CELERY_BROKER_TRANSPORT_OPTIONS')) > 0:
    CELERY_BROKER_TRANSPORT_OPTIONS = json.loads(env('DD_CELERY_BROKER_TRANSPORT_OPTIONS'))
CELERY_IMPORTS = ('dojo.tools.tool_issue_updater',)
CELERY_BEAT_SCHEDULE = {'add-alerts': {'task': 'dojo.tasks.add_alerts', 'schedule': timedelta(hours=1), 'args': [timedelta(hours=1)]}, 'cleanup-alerts': {'task': 'dojo.tasks.cleanup_alerts', 'schedule': timedelta(hours=8)}, 'dedupe-delete': {'task': 'dojo.tasks.async_dupe_delete', 'schedule': timedelta(minutes=1), 'args': [timedelta(minutes=1)]}, 'update-findings-from-source-issues': {'task': 'dojo.tools.tool_issue_updater.update_findings_from_source_issues', 'schedule': timedelta(hours=3)}, 'compute-sla-age-and-notify': {'task': 'dojo.tasks.async_sla_compute_and_notify_task', 'schedule': crontab(hour=7, minute=30)}, 'risk_acceptance_expiration_handler': {'task': 'dojo.risk_acceptance.helper.expiration_handler', 'schedule': crontab(minute=0, hour='*/3')}}
PROMETHEUS_EXPORT_MIGRATIONS = False
if env('DD_DJANGO_METRICS_ENABLED'):
    DJANGO_METRICS_ENABLED = env('DD_DJANGO_METRICS_ENABLED')
    INSTALLED_APPS = INSTALLED_APPS + ('django_prometheus',)
    MIDDLEWARE = ['django_prometheus.middleware.PrometheusBeforeMiddleware'] + MIDDLEWARE + ['django_prometheus.middleware.PrometheusAfterMiddleware']
    database_engine = DATABASES.get('default').get('ENGINE')
    DATABASES['default']['ENGINE'] = database_engine.replace('django.', 'django_prometheus.', 1)
    LOGIN_EXEMPT_URLS += ('^%sdjango_metrics/' % URL_PREFIX,)
HASHCODE_FIELDS_PER_SCANNER = {'Anchore Engine Scan': ['title', 'severity', 'component_name', 'component_version', 'file_path'], 'AnchoreCTL Vuln Report': ['title', 'severity', 'component_name', 'component_version', 'file_path'], 'AnchoreCTL Policies Report': ['title', 'severity', 'component_name', 'file_path'], 'Anchore Enterprise Policy Check': ['title', 'severity', 'component_name', 'file_path'], 'Anchore Grype': ['title', 'severity', 'component_name', 'component_version'], 'Aqua Scan': ['severity', 'vulnerability_ids', 'component_name', 'component_version'], 'Bandit Scan': ['file_path', 'line', 'vuln_id_from_tool'], 'CargoAudit Scan': ['vulnerability_ids', 'severity', 'component_name', 'component_version', 'vuln_id_from_tool'], 'Checkmarx Scan': ['cwe', 'severity', 'file_path'], 'Checkmarx OSA': ['vulnerability_ids', 'component_name'], 'Cloudsploit Scan': ['title', 'description'], 'SonarQube Scan': ['cwe', 'severity', 'file_path'], 'SonarQube API Import': ['title', 'file_path', 'line'], 'Dependency Check Scan': ['title', 'cwe', 'file_path'], 'Dockle Scan': ['title', 'description', 'vuln_id_from_tool'], 'Dependency Track Finding Packaging Format (FPF) Export': ['component_name', 'component_version', 'vulnerability_ids'], 'Mobsfscan Scan': ['title', 'severity', 'cwe'], 'Tenable Scan': ['title', 'severity', 'vulnerability_ids', 'cwe'], 'Nexpose Scan': ['title', 'severity', 'vulnerability_ids', 'cwe'], 'NPM Audit Scan': ['title', 'severity', 'file_path', 'vulnerability_ids', 'cwe'], 'Yarn Audit Scan': ['title', 'severity', 'file_path', 'vulnerability_ids', 'cwe'], 'Whitesource Scan': ['title', 'severity', 'description'], 'ZAP Scan': ['title', 'cwe', 'severity'], 'Qualys Scan': ['title', 'severity', 'endpoints'], 'PHP Symfony Security Check': ['title', 'vulnerability_ids'], 'Clair Scan': ['title', 'vulnerability_ids', 'description', 'severity'], 'Clair Klar Scan': ['title', 'description', 'severity'], 'Symfony Security Check': ['title', 'vulnerability_ids'], 'DSOP Scan': ['vulnerability_ids'], 'Acunetix Scan': ['title', 'description'], 'Acunetix360 Scan': ['title', 'description'], 'Terrascan Scan': ['vuln_id_from_tool', 'title', 'severity', 'file_path', 'line', 'component_name'], 'Trivy Operator Scan': ['title', 'severity', 'vulnerability_ids'], 'Trivy Scan': ['title', 'severity', 'vulnerability_ids', 'cwe'], 'TFSec Scan': ['severity', 'vuln_id_from_tool', 'file_path', 'line'], 'Snyk Scan': ['vuln_id_from_tool', 'file_path', 'component_name', 'component_version'], 'GitLab Dependency Scanning Report': ['title', 'vulnerability_ids', 'file_path', 'component_name', 'component_version'], 'SpotBugs Scan': ['cwe', 'severity', 'file_path', 'line'], 'JFrog Xray Unified Scan': ['vulnerability_ids', 'file_path', 'component_name', 'component_version'], 'Scout Suite Scan': ['file_path', 'vuln_id_from_tool'], 'AWS Security Hub Scan': ['unique_id_from_tool'], 'Meterian Scan': ['cwe', 'component_name', 'component_version', 'description', 'severity'], 'Govulncheck Scanner': ['unique_id_from_tool'], 'Github Vulnerability Scan': ['title', 'severity', 'component_name', 'vulnerability_ids', 'file_path'], 'Azure Security Center Recommendations Scan': ['unique_id_from_tool'], 'Solar Appscreener Scan': ['title', 'file_path', 'line', 'severity'], 'pip-audit Scan': ['vuln_id_from_tool', 'component_name', 'component_version'], 'Edgescan Scan': ['unique_id_from_tool'], 'Bugcrowd API Import': ['unique_id_from_tool'], 'Rubocop Scan': ['vuln_id_from_tool', 'file_path', 'line'], 'JFrog Xray Scan': ['title', 'description', 'component_name', 'component_version'], 'CycloneDX Scan': ['vuln_id_from_tool', 'component_name', 'component_version'], 'SSLyze Scan (JSON)': ['title', 'description'], 'Harbor Vulnerability Scan': ['title', 'mitigation'], 'Rusty Hog Scan': ['file_path', 'payload'], 'StackHawk HawkScan': ['vuln_id_from_tool', 'component_name', 'component_version'], 'Hydra Scan': ['title', 'description'], 'DrHeader JSON Importer': ['title', 'description'], 'PWN SAST': ['title', 'description'], 'Whispers': ['vuln_id_from_tool', 'file_path', 'line'], 'Blackduck Hub Scan': ['title', 'vulnerability_ids', 'component_name', 'component_version'], 'BlackDuck API': ['unique_id_from_tool'], 'docker-bench-security Scan': ['unique_id_from_tool'], 'Veracode SourceClear Scan': ['title', 'vulnerability_ids', 'component_name', 'component_version', 'severity'], 'Vulners Scan': ['vuln_id_from_tool', 'component_name'], 'Twistlock Image Scan': ['title', 'severity', 'component_name', 'component_version'], 'NeuVector (REST)': ['title', 'severity', 'component_name', 'component_version'], 'NeuVector (compliance)': ['title', 'vuln_id_from_tool', 'description'], 'Wpscan': ['title', 'description', 'severity'], 'Codechecker Report native': ['unique_id_from_tool'], 'Popeye Scan': ['title', 'description'], 'Wazuh Scan': ['title'], 'Nuclei Scan': ['title', 'cwe', 'severity'], 'KubeHunter Scan': ['title', 'description'], 'kube-bench Scan': ['title', 'vuln_id_from_tool', 'description'], 'Threagile risks report': ['title', 'cwe', 'severity']}
if len(env('DD_HASHCODE_FIELDS_PER_SCANNER')) > 0:
    env_hashcode_fields_per_scanner = json.loads(env('DD_HASHCODE_FIELDS_PER_SCANNER'))
    for (key, value) in env_hashcode_fields_per_scanner.items():
        if key in HASHCODE_FIELDS_PER_SCANNER:
            print('Replacing {} with value {} from env var DD_HASHCODE_FIELDS_PER_SCANNER'.format(key, value))
            HASHCODE_FIELDS_PER_SCANNER[key] = value
HASHCODE_ALLOWS_NULL_CWE = {'Anchore Engine Scan': True, 'AnchoreCTL Vuln Report': True, 'AnchoreCTL Policies Report': True, 'Anchore Enterprise Policy Check': True, 'Anchore Grype': True, 'AWS Prowler Scan': True, 'AWS Prowler V3': True, 'Checkmarx Scan': False, 'Checkmarx OSA': True, 'Cloudsploit Scan': True, 'SonarQube Scan': False, 'Dependency Check Scan': True, 'Mobsfscan Scan': False, 'Tenable Scan': True, 'Nexpose Scan': True, 'NPM Audit Scan': True, 'Yarn Audit Scan': True, 'Whitesource Scan': True, 'ZAP Scan': False, 'Qualys Scan': True, 'DSOP Scan': True, 'Acunetix Scan': True, 'Acunetix360 Scan': True, 'Trivy Operator Scan': True, 'Trivy Scan': True, 'SpotBugs Scan': False, 'Scout Suite Scan': True, 'AWS Security Hub Scan': True, 'Meterian Scan': True, 'SARIF': True, 'Hadolint Dockerfile check': True, 'Semgrep JSON Report': True, 'Generic Findings Import': True, 'Edgescan Scan': True, 'Bugcrowd API Import': True, 'Veracode SourceClear Scan': True, 'Vulners Scan': True, 'Twistlock Image Scan': True, 'Wpscan': True, 'Rusty Hog Scan': True, 'Codechecker Report native': True, 'Wazuh': True, 'Nuclei Scan': True, 'Threagile risks report': True}
HASHCODE_ALLOWED_FIELDS = ['title', 'cwe', 'vulnerability_ids', 'line', 'file_path', 'payload', 'component_name', 'component_version', 'description', 'endpoints', 'unique_id_from_tool', 'severity', 'vuln_id_from_tool', 'mitigation']
HASH_CODE_FIELDS_ALWAYS = ['service']
DEDUPE_ALGO_LEGACY = 'legacy'
DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL = 'unique_id_from_tool'
DEDUPE_ALGO_HASH_CODE = 'hash_code'
DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL_OR_HASH_CODE = 'unique_id_from_tool_or_hash_code'
DEDUPE_ALGO_ENDPOINT_FIELDS = ['host', 'path']
DEDUPLICATION_ALGORITHM_PER_PARSER = {'Anchore Engine Scan': DEDUPE_ALGO_HASH_CODE, 'AnchoreCTL Vuln Report': DEDUPE_ALGO_HASH_CODE, 'AnchoreCTL Policies Report': DEDUPE_ALGO_HASH_CODE, 'Anchore Enterprise Policy Check': DEDUPE_ALGO_HASH_CODE, 'Anchore Grype': DEDUPE_ALGO_HASH_CODE, 'Aqua Scan': DEDUPE_ALGO_HASH_CODE, 'AuditJS Scan': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'AWS Prowler Scan': DEDUPE_ALGO_HASH_CODE, 'AWS Prowler V3': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'AWS Security Finding Format (ASFF) Scan': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Burp REST API': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Bandit Scan': DEDUPE_ALGO_HASH_CODE, 'CargoAudit Scan': DEDUPE_ALGO_HASH_CODE, 'Checkmarx Scan detailed': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Checkmarx Scan': DEDUPE_ALGO_HASH_CODE, 'Checkmarx OSA': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL_OR_HASH_CODE, 'Codechecker Report native': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Coverity API': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Cobalt.io API': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Dependency Track Finding Packaging Format (FPF) Export': DEDUPE_ALGO_HASH_CODE, 'Mobsfscan Scan': DEDUPE_ALGO_HASH_CODE, 'SonarQube Scan detailed': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'SonarQube Scan': DEDUPE_ALGO_HASH_CODE, 'SonarQube API Import': DEDUPE_ALGO_HASH_CODE, 'Dependency Check Scan': DEDUPE_ALGO_HASH_CODE, 'Dockle Scan': DEDUPE_ALGO_HASH_CODE, 'Tenable Scan': DEDUPE_ALGO_HASH_CODE, 'Nexpose Scan': DEDUPE_ALGO_HASH_CODE, 'NPM Audit Scan': DEDUPE_ALGO_HASH_CODE, 'Yarn Audit Scan': DEDUPE_ALGO_HASH_CODE, 'Whitesource Scan': DEDUPE_ALGO_HASH_CODE, 'ZAP Scan': DEDUPE_ALGO_HASH_CODE, 'Qualys Scan': DEDUPE_ALGO_HASH_CODE, 'PHP Symfony Security Check': DEDUPE_ALGO_HASH_CODE, 'Acunetix Scan': DEDUPE_ALGO_HASH_CODE, 'Acunetix360 Scan': DEDUPE_ALGO_HASH_CODE, 'Clair Scan': DEDUPE_ALGO_HASH_CODE, 'Clair Klar Scan': DEDUPE_ALGO_HASH_CODE, 'Veracode Scan': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL_OR_HASH_CODE, 'Veracode SourceClear Scan': DEDUPE_ALGO_HASH_CODE, 'Symfony Security Check': DEDUPE_ALGO_HASH_CODE, 'DSOP Scan': DEDUPE_ALGO_HASH_CODE, 'Terrascan Scan': DEDUPE_ALGO_HASH_CODE, 'Trivy Operator Scan': DEDUPE_ALGO_HASH_CODE, 'Trivy Scan': DEDUPE_ALGO_HASH_CODE, 'TFSec Scan': DEDUPE_ALGO_HASH_CODE, 'HackerOne Cases': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL_OR_HASH_CODE, 'Snyk Scan': DEDUPE_ALGO_HASH_CODE, 'GitLab Dependency Scanning Report': DEDUPE_ALGO_HASH_CODE, 'GitLab SAST Report': DEDUPE_ALGO_HASH_CODE, 'Govulncheck Scanner': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'GitLab Container Scan': DEDUPE_ALGO_HASH_CODE, 'GitLab Secret Detection Report': DEDUPE_ALGO_HASH_CODE, 'Checkov Scan': DEDUPE_ALGO_HASH_CODE, 'SpotBugs Scan': DEDUPE_ALGO_HASH_CODE, 'JFrog Xray Unified Scan': DEDUPE_ALGO_HASH_CODE, 'Scout Suite Scan': DEDUPE_ALGO_HASH_CODE, 'AWS Security Hub Scan': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Meterian Scan': DEDUPE_ALGO_HASH_CODE, 'Github Vulnerability Scan': DEDUPE_ALGO_HASH_CODE, 'Cloudsploit Scan': DEDUPE_ALGO_HASH_CODE, 'KICS Scan': DEDUPE_ALGO_HASH_CODE, 'SARIF': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL_OR_HASH_CODE, 'Azure Security Center Recommendations Scan': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Hadolint Dockerfile check': DEDUPE_ALGO_HASH_CODE, 'Semgrep JSON Report': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL_OR_HASH_CODE, 'Generic Findings Import': DEDUPE_ALGO_HASH_CODE, 'Trufflehog3 Scan': DEDUPE_ALGO_HASH_CODE, 'Detect-secrets Scan': DEDUPE_ALGO_HASH_CODE, 'Solar Appscreener Scan': DEDUPE_ALGO_HASH_CODE, 'Gitleaks Scan': DEDUPE_ALGO_HASH_CODE, 'pip-audit Scan': DEDUPE_ALGO_HASH_CODE, 'Edgescan Scan': DEDUPE_ALGO_HASH_CODE, 'Bugcrowd API': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'Rubocop Scan': DEDUPE_ALGO_HASH_CODE, 'JFrog Xray Scan': DEDUPE_ALGO_HASH_CODE, 'CycloneDX Scan': DEDUPE_ALGO_HASH_CODE, 'SSLyze Scan (JSON)': DEDUPE_ALGO_HASH_CODE, 'Harbor Vulnerability Scan': DEDUPE_ALGO_HASH_CODE, 'Rusty Hog Scan': DEDUPE_ALGO_HASH_CODE, 'StackHawk HawkScan': DEDUPE_ALGO_HASH_CODE, 'Hydra Scan': DEDUPE_ALGO_HASH_CODE, 'DrHeader JSON Importer': DEDUPE_ALGO_HASH_CODE, 'PWN SAST': DEDUPE_ALGO_HASH_CODE, 'Whispers': DEDUPE_ALGO_HASH_CODE, 'Blackduck Hub Scan': DEDUPE_ALGO_HASH_CODE, 'BlackDuck API': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL, 'docker-bench-security Scan': DEDUPE_ALGO_HASH_CODE, 'Vulners Scan': DEDUPE_ALGO_HASH_CODE, 'Twistlock Image Scan': DEDUPE_ALGO_HASH_CODE, 'NeuVector (REST)': DEDUPE_ALGO_HASH_CODE, 'NeuVector (compliance)': DEDUPE_ALGO_HASH_CODE, 'Wpscan': DEDUPE_ALGO_HASH_CODE, 'Popeye Scan': DEDUPE_ALGO_HASH_CODE, 'Nuclei Scan': DEDUPE_ALGO_HASH_CODE, 'KubeHunter Scan': DEDUPE_ALGO_HASH_CODE, 'kube-bench Scan': DEDUPE_ALGO_HASH_CODE, 'Threagile risks report': DEDUPE_ALGO_UNIQUE_ID_FROM_TOOL_OR_HASH_CODE}
if len(env('DD_DEDUPLICATION_ALGORITHM_PER_PARSER')) > 0:
    env_dedup_algorithm_per_parser = json.loads(env('DD_DEDUPLICATION_ALGORITHM_PER_PARSER'))
    for (key, value) in env_dedup_algorithm_per_parser.items():
        if key in DEDUPLICATION_ALGORITHM_PER_PARSER:
            print('Replacing {} with value {} from env var DD_DEDUPLICATION_ALGORITHM_PER_PARSER'.format(key, value))
            DEDUPLICATION_ALGORITHM_PER_PARSER[key] = value
DUPE_DELETE_MAX_PER_RUN = env('DD_DUPE_DELETE_MAX_PER_RUN')
DISABLE_FINDING_MERGE = env('DD_DISABLE_FINDING_MERGE')
TRACK_IMPORT_HISTORY = env('DD_TRACK_IMPORT_HISTORY')
JIRA_ISSUE_TYPE_CHOICES_CONFIG = (('Task', 'Task'), ('Story', 'Story'), ('Epic', 'Epic'), ('Spike', 'Spike'), ('Bug', 'Bug'), ('Security', 'Security'))
if env('DD_JIRA_EXTRA_ISSUE_TYPES') != '':
    if env('DD_JIRA_EXTRA_ISSUE_TYPES').count(',') > 0:
        for extra_type in env('DD_JIRA_EXTRA_ISSUE_TYPES').split(','):
            JIRA_ISSUE_TYPE_CHOICES_CONFIG += ((extra_type, extra_type),)
    else:
        JIRA_ISSUE_TYPE_CHOICES_CONFIG += ((env('DD_JIRA_EXTRA_ISSUE_TYPES'), env('DD_JIRA_EXTRA_ISSUE_TYPES')),)
JIRA_SSL_VERIFY = env('DD_JIRA_SSL_VERIFY')
LOGGING_HANDLER = env('DD_LOGGING_HANDLER')
LOG_LEVEL = env('DD_LOG_LEVEL')
if not LOG_LEVEL:
    LOG_LEVEL = 'DEBUG' if DEBUG else 'INFO'
LOGGING = {'version': 1, 'disable_existing_loggers': False, 'formatters': {'verbose': {'format': '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s', 'datefmt': '%d/%b/%Y %H:%M:%S'}, 'simple': {'format': '%(levelname)s %(funcName)s %(lineno)d %(message)s'}, 'json': {'()': 'json_log_formatter.JSONFormatter'}}, 'filters': {'require_debug_false': {'()': 'django.utils.log.RequireDebugFalse'}, 'require_debug_true': {'()': 'django.utils.log.RequireDebugTrue'}}, 'handlers': {'mail_admins': {'level': 'ERROR', 'filters': ['require_debug_false'], 'class': 'django.utils.log.AdminEmailHandler'}, 'console': {'class': 'logging.StreamHandler', 'formatter': 'verbose'}, 'json_console': {'class': 'logging.StreamHandler', 'formatter': 'json'}}, 'loggers': {'django.request': {'handlers': ['mail_admins', 'console'], 'level': '%s' % LOG_LEVEL, 'propagate': False}, 'django.security': {'handlers': ['%s' % LOGGING_HANDLER], 'level': '%s' % LOG_LEVEL, 'propagate': False}, 'celery': {'handlers': ['%s' % LOGGING_HANDLER], 'level': '%s' % LOG_LEVEL, 'propagate': False, 'worker_hijack_root_logger': False}, 'dojo': {'handlers': ['%s' % LOGGING_HANDLER], 'level': '%s' % LOG_LEVEL, 'propagate': False}, 'dojo.specific-loggers.deduplication': {'handlers': ['%s' % LOGGING_HANDLER], 'level': '%s' % LOG_LEVEL, 'propagate': False}, 'saml2': {'handlers': ['%s' % LOGGING_HANDLER], 'level': '%s' % LOG_LEVEL, 'propagate': False}, 'MARKDOWN': {'handlers': ['%s' % LOGGING_HANDLER], 'level': '%s' % LOG_LEVEL, 'propagate': False}, 'titlecase': {'handlers': ['%s' % LOGGING_HANDLER], 'level': '%s' % LOG_LEVEL, 'propagate': False}}}
DEFAULT_EXCEPTION_REPORTER_FILTER = 'dojo.settings.exception_filter.CustomExceptionReporterFilter'
SILENCED_SYSTEM_CHECKS = ['mysql.E001']
DATA_UPLOAD_MAX_NUMBER_FIELDS = 10240
SCAN_FILE_MAX_SIZE = env('DD_SCAN_FILE_MAX_SIZE')
QUALYS_WAS_WEAKNESS_IS_VULN = env('DD_QUALYS_WAS_WEAKNESS_IS_VULN')
QUALYS_WAS_UNIQUE_ID = False
PARSER_EXCLUDE = env('DD_PARSER_EXCLUDE')
SERIALIZATION_MODULES = {'xml': 'tagulous.serializers.xml_serializer', 'json': 'tagulous.serializers.json', 'python': 'tagulous.serializers.python', 'yaml': 'tagulous.serializers.pyyaml'}
TAGULOUS_AUTOCOMPLETE_JS = ('tagulous/lib/select2-4/js/select2.full.min.js', 'tagulous/tagulous.js', 'tagulous/adaptor/select2-4.js')
TAGULOUS_AUTOCOMPLETE_SETTINGS = {'placeholder': 'Enter some tags (comma separated, use enter to select / create a new tag)', 'width': '70%'}
EDITABLE_MITIGATED_DATA = env('DD_EDITABLE_MITIGATED_DATA')
USE_L10N = True
FEATURE_FINDING_GROUPS = env('DD_FEATURE_FINDING_GROUPS')
JIRA_TEMPLATE_ROOT = env('DD_JIRA_TEMPLATE_ROOT')
TEMPLATE_DIR_PREFIX = env('DD_TEMPLATE_DIR_PREFIX')
DUPLICATE_CLUSTER_CASCADE_DELETE = env('DD_DUPLICATE_CLUSTER_CASCADE_DELETE')
SONARQUBE_API_PARSER_HOTSPOTS = env('DD_SONARQUBE_API_PARSER_HOTSPOTS')
ASYNC_FINDING_IMPORT = env('DD_ASYNC_FINDING_IMPORT')
ASYNC_FINDING_IMPORT_CHUNK_SIZE = env('DD_ASYNC_FINDING_IMPORT_CHUNK_SIZE')
ASYNC_OBJECT_DELETE = env('DD_ASYNC_OBJECT_DELETE')
ASYNC_OBEJECT_DELETE_CHUNK_SIZE = env('DD_ASYNC_OBEJECT_DELETE_CHUNK_SIZE')
DELETE_PREVIEW = env('DD_DELETE_PREVIEW')
SILENCED_SYSTEM_CHECKS = ['django_jsonfield_backport.W001']
VULNERABILITY_URLS = {'CVE': 'https://nvd.nist.gov/vuln/detail/', 'GHSA': 'https://github.com/advisories/', 'OSV': 'https://osv.dev/vulnerability/', 'PYSEC': 'https://osv.dev/vulnerability/', 'SNYK': 'https://snyk.io/vuln/', 'RUSTSEC': 'https://rustsec.org/advisories/', 'VNS': 'https://vulners.com/'}
FILE_UPLOAD_TYPES = env('DD_FILE_UPLOAD_TYPES')
AUDITLOG_DISABLE_ON_RAW_SAVE = False
ADDITIONAL_HEADERS = env('DD_ADDITIONAL_HEADERS')
CREATE_CLOUD_BANNER = env('DD_CREATE_CLOUD_BANNER')