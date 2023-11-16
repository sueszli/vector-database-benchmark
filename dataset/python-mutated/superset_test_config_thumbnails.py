from copy import copy
from superset.config import *
AUTH_USER_REGISTRATION_ROLE = 'alpha'
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(DATA_DIR, 'unittests.integration_tests.db')
DEBUG = True
SUPERSET_WEBSERVER_PORT = 8081
if 'SUPERSET__SQLALCHEMY_DATABASE_URI' in os.environ:
    SQLALCHEMY_DATABASE_URI = os.environ['SUPERSET__SQLALCHEMY_DATABASE_URI']
if 'sqlite' in SQLALCHEMY_DATABASE_URI:
    logger.warning('SQLite Database support for metadata databases will be removed         in a future version of Superset.')
SQL_SELECT_AS_CTA = True
SQL_MAX_ROW = 666

def GET_FEATURE_FLAGS_FUNC(ff):
    if False:
        for i in range(10):
            print('nop')
    ff_copy = copy(ff)
    ff_copy['super'] = 'set'
    return ff_copy
TESTING = True
WTF_CSRF_ENABLED = False
PUBLIC_ROLE_LIKE = 'Gamma'
AUTH_ROLE_PUBLIC = 'Public'
EMAIL_NOTIFICATIONS = False
CACHE_CONFIG = {'CACHE_TYPE': 'SimpleCache'}
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
REDIS_CELERY_DB = os.environ.get('REDIS_CELERY_DB', 2)
REDIS_RESULTS_DB = os.environ.get('REDIS_RESULTS_DB', 3)

class CeleryConfig:
    broker_url = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_CELERY_DB}'
    imports = ('superset.sql_lab', 'superset.tasks.thumbnails')
    concurrency = 1
CELERY_CONFIG = CeleryConfig
FEATURE_FLAGS = {'foo': 'bar', 'KV_STORE': False, 'SHARE_QUERIES_VIA_KV_STORE': False, 'THUMBNAILS': True, 'THUMBNAILS_SQLA_LISTENERS': False}
THUMBNAIL_CACHE_CONFIG = {'CACHE_TYPE': 'RedisCache', 'CACHE_DEFAULT_TIMEOUT': 10000, 'CACHE_KEY_PREFIX': 'superset_thumbnails_', 'CACHE_REDIS_HOST': REDIS_HOST, 'CACHE_REDIS_PORT': REDIS_PORT, 'CACHE_REDIS_DB': REDIS_CELERY_DB}