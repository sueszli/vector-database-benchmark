"""
Module contain database set up and teardown related functionality.
"""
from __future__ import absolute_import
from oslo_config import cfg
from st2common.models import db
from st2common.persistence import db_init
__all__ = ['db_config', 'db_setup', 'db_teardown']

def db_config():
    if False:
        return 10
    username = getattr(cfg.CONF.database, 'username', None)
    password = getattr(cfg.CONF.database, 'password', None)
    return {'db_name': cfg.CONF.database.db_name, 'db_host': cfg.CONF.database.host, 'db_port': cfg.CONF.database.port, 'username': username, 'password': password, 'ssl': cfg.CONF.database.ssl, 'ssl_keyfile': cfg.CONF.database.ssl_keyfile, 'ssl_certfile': cfg.CONF.database.ssl_certfile, 'ssl_cert_reqs': cfg.CONF.database.ssl_cert_reqs, 'ssl_ca_certs': cfg.CONF.database.ssl_ca_certs, 'authentication_mechanism': cfg.CONF.database.authentication_mechanism, 'ssl_match_hostname': cfg.CONF.database.ssl_match_hostname}

def db_setup(ensure_indexes=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates the database and indexes (optional).\n    '
    db_cfg = db_config()
    db_cfg['ensure_indexes'] = ensure_indexes
    connection = db_init.db_setup_with_retry(**db_cfg)
    return connection

def db_teardown():
    if False:
        for i in range(10):
            print('nop')
    '\n    Disconnects from the database.\n    '
    return db.db_teardown()