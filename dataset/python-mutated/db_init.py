from __future__ import absolute_import
import mongoengine
from oslo_config import cfg
import retrying
from st2common import log as logging
from st2common.models.db import db_setup
__all__ = ['db_setup_with_retry']
LOG = logging.getLogger(__name__)

def _retry_if_connection_error(error):
    if False:
        i = 10
        return i + 15
    is_connection_error = isinstance(error, mongoengine.connection.ConnectionFailure)
    if is_connection_error:
        LOG.warn('Retry on ConnectionError - %s', error)
    return is_connection_error

def db_func_with_retry(db_func, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    This method is a generic retry function to support database setup and cleanup.\n    '
    retrying_obj = retrying.Retrying(retry_on_exception=_retry_if_connection_error, wait_exponential_multiplier=cfg.CONF.database.connection_retry_backoff_mul * 1000, wait_exponential_max=cfg.CONF.database.connection_retry_backoff_max_s * 1000, stop_max_delay=cfg.CONF.database.connection_retry_max_delay_m * 60 * 1000)
    return retrying_obj.call(db_func, *args, **kwargs)

def db_setup_with_retry(db_name, db_host, db_port, username=None, password=None, ensure_indexes=True, ssl=False, ssl_keyfile=None, ssl_certfile=None, ssl_cert_reqs=None, ssl_ca_certs=None, authentication_mechanism=None, ssl_match_hostname=True):
    if False:
        print('Hello World!')
    '\n    This method is a retry version of db_setup.\n    '
    return db_func_with_retry(db_setup, db_name, db_host, db_port, username=username, password=password, ensure_indexes=ensure_indexes, ssl=ssl, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile, ssl_cert_reqs=ssl_cert_reqs, ssl_ca_certs=ssl_ca_certs, authentication_mechanism=authentication_mechanism, ssl_match_hostname=ssl_match_hostname)