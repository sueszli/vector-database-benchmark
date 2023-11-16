import os
import socket
from .base import CommunityBaseSettings

class DockerBaseSettings(CommunityBaseSettings):
    """Settings for local development with Docker"""
    DEBUG = bool(os.environ.get('RTD_DJANGO_DEBUG', True))
    DOCKER_ENABLE = True
    RTD_DOCKER_COMPOSE = True
    RTD_DOCKER_COMPOSE_VOLUME = 'community_build-user-builds'
    RTD_DOCKER_USER = f'{os.geteuid()}:{os.getegid()}'
    DOCKER_LIMITS = {'memory': '1g', 'time': 900}
    PRODUCTION_DOMAIN = os.environ.get('RTD_PRODUCTION_DOMAIN', 'devthedocs.org')
    PUBLIC_DOMAIN = os.environ.get('RTD_PUBLIC_DOMAIN', 'devthedocs.org')
    PUBLIC_API_URL = f'http://{PRODUCTION_DOMAIN}'
    SLUMBER_API_HOST = 'http://web:8000'
    RTD_EXTERNAL_VERSION_DOMAIN = 'build.devthedocs.org'
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    STATIC_URL = '/static/'
    USE_X_FORWARDED_HOST = True
    HOSTIP = os.environ.get('HOSTIP')
    (_, __, ips) = socket.gethostbyname_ex(socket.gethostname())
    if ips and (not HOSTIP):
        HOSTIP = ips[0][:-1] + '1'
    USE_PROMOS = False
    ADSERVER_API_BASE = f'http://{HOSTIP}:5000'
    ADSERVER_API_KEY = None
    ADSERVER_API_TIMEOUT = 2

    @property
    def DOCROOT(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(super().DOCROOT, socket.gethostname())

    @property
    def RTD_EXT_THEME_DEV_SERVER_ENABLED(self):
        if False:
            for i in range(10):
                print('nop')
        return os.environ.get('RTD_EXT_THEME_DEV_SERVER_ENABLED') is not None

    @property
    def RTD_EXT_THEME_DEV_SERVER(self):
        if False:
            while True:
                i = 10
        if self.RTD_EXT_THEME_DEV_SERVER_ENABLED:
            return 'http://assets.devthedocs.org:10001'
    ELASTICSEARCH_DSL_AUTOSYNC = 'SEARCH' in os.environ
    RTD_CLEAN_AFTER_BUILD = True
    AUTH_PASSWORD_VALIDATORS = []

    @property
    def RTD_EMBED_API_EXTERNAL_DOMAINS(self):
        if False:
            return 10
        domains = super().RTD_EMBED_API_EXTERNAL_DOMAINS
        domains.extend(['.*\\.readthedocs\\.io', '.*\\.org\\.readthedocs\\.build', '.*\\.readthedocs-hosted\\.com', '.*\\.com\\.readthedocs\\.build'])
        return domains

    @property
    def LOGGING(self):
        if False:
            return 10
        logging = super().LOGGING
        logging['handlers']['console']['level'] = os.environ.get('RTD_LOGGING_LEVEL', 'INFO')
        logging['formatters']['default']['format'] = '[%(asctime)s] ' + self.LOG_FORMAT
        logging['disable_existing_loggers'] = False
        logging['handlers']['console']['formatter'] = 'colored_console'
        logging['loggers'].update({'django.server': {'handlers': ['null'], 'propagate': False}, 'boto3': {'handlers': ['null'], 'propagate': False}, 'botocore': {'handlers': ['null'], 'propagate': False}, 's3transfer': {'handlers': ['null'], 'propagate': False}, 'urllib3': {'handlers': ['null'], 'propagate': False}, 'git.cmd': {'handlers': ['null'], 'propagate': False}})
        return logging

    @property
    def DATABASES(self):
        if False:
            for i in range(10):
                print('nop')
        return {'default': {'ENGINE': 'django.db.backends.postgresql_psycopg2', 'NAME': 'docs_db', 'USER': os.environ.get('DB_USER', 'docs_user'), 'PASSWORD': os.environ.get('DB_PWD', 'docs_pwd'), 'HOST': os.environ.get('DB_HOST', 'database'), 'PORT': ''}, 'telemetry': {'ENGINE': 'django.db.backends.postgresql_psycopg2', 'NAME': 'telemetry', 'USER': os.environ.get('DB_USER', 'docs_user'), 'PASSWORD': os.environ.get('DB_PWD', 'docs_pwd'), 'HOST': os.environ.get('DB_HOST', 'database'), 'PORT': ''}}
    ACCOUNT_EMAIL_VERIFICATION = 'none'
    SESSION_COOKIE_DOMAIN = None
    CACHES = {'default': {'BACKEND': 'django.core.cache.backends.redis.RedisCache', 'LOCATION': 'redis://:redispassword@cache:6379'}}
    CACHEOPS_REDIS = f'redis://:redispassword@cache:6379/1'
    BROKER_URL = f'redis://:redispassword@cache:6379/0'
    CELERY_ALWAYS_EAGER = False
    EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
    RTD_BUILD_MEDIA_STORAGE = 'readthedocs.storage.s3_storage.S3BuildMediaStorage'
    RTD_BUILD_ENVIRONMENT_STORAGE = 'readthedocs.storage.s3_storage.S3BuildEnvironmentStorage'
    RTD_BUILD_TOOLS_STORAGE = 'readthedocs.storage.s3_storage.S3BuildToolsStorage'
    STATICFILES_STORAGE = 'readthedocs.storage.s3_storage.S3StaticStorage'
    RTD_STATICFILES_STORAGE = 'readthedocs.storage.s3_storage.NoManifestS3StaticStorage'
    AWS_ACCESS_KEY_ID = 'admin'
    AWS_SECRET_ACCESS_KEY = 'password'
    S3_MEDIA_STORAGE_BUCKET = 'media'
    S3_BUILD_COMMANDS_STORAGE_BUCKET = 'builds'
    S3_BUILD_ENVIRONMENT_STORAGE_BUCKET = 'envs'
    S3_BUILD_TOOLS_STORAGE_BUCKET = 'build-tools'
    S3_STATIC_STORAGE_BUCKET = 'static'
    S3_STATIC_STORAGE_OVERRIDE_HOSTNAME = PRODUCTION_DOMAIN
    S3_MEDIA_STORAGE_OVERRIDE_HOSTNAME = PRODUCTION_DOMAIN
    AWS_S3_ENCRYPTION = False
    AWS_S3_SECURE_URLS = False
    AWS_S3_USE_SSL = False
    AWS_S3_ENDPOINT_URL = 'http://storage:9000/'
    AWS_QUERYSTRING_AUTH = False
    RTD_SAVE_BUILD_COMMANDS_TO_STORAGE = True
    RTD_BUILD_COMMANDS_STORAGE = 'readthedocs.storage.s3_storage.S3BuildCommandsStorage'
    BUILD_COLD_STORAGE_URL = 'http://storage:9000/builds'
    STATICFILES_DIRS = [os.path.join(CommunityBaseSettings.SITE_ROOT, 'readthedocs', 'static'), os.path.join(CommunityBaseSettings.SITE_ROOT, 'media')]
    DATA_UPLOAD_MAX_NUMBER_FIELDS = None
DockerBaseSettings.load_settings(__name__)