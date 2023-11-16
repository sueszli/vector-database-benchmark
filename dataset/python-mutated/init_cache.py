from __future__ import annotations
import hashlib
from tempfile import gettempdir
from flask_caching import Cache
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException
HASH_METHOD_MAPPING = {'md5': hashlib.md5, 'sha1': hashlib.sha1, 'sha224': hashlib.sha224, 'sha256': hashlib.sha256, 'sha384': hashlib.sha384, 'sha512': hashlib.sha512}

def init_cache(app):
    if False:
        for i in range(10):
            print('nop')
    webserver_caching_hash_method = conf.get(section='webserver', key='CACHING_HASH_METHOD', fallback='md5').casefold()
    cache_config = {'CACHE_TYPE': 'flask_caching.backends.filesystem', 'CACHE_DIR': gettempdir()}
    mapped_hash_method = HASH_METHOD_MAPPING.get(webserver_caching_hash_method)
    if mapped_hash_method is None:
        raise AirflowConfigException(f'Unsupported webserver caching hash method: `{webserver_caching_hash_method}`.')
    cache_config['CACHE_OPTIONS'] = {'hash_method': mapped_hash_method}
    Cache(app=app, config=cache_config)