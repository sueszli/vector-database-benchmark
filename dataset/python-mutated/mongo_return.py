"""
Return data to a mongodb server

Required python modules: pymongo


This returner will send data from the minions to a MongoDB server. To
configure the settings for your MongoDB server, add the following lines
to the minion config files.

.. code-block:: yaml

    mongo.db: <database name>
    mongo.host: <server ip address>
    mongo.user: <MongoDB username>
    mongo.password: <MongoDB user password>
    mongo.port: 27017

Alternative configuration values can be used by prefacing the configuration.
Any values not found in the alternative configuration will be pulled from
the default location.

.. code-block:: yaml

    alternative.mongo.db: <database name>
    alternative.mongo.host: <server ip address>
    alternative.mongo.user: <MongoDB username>
    alternative.mongo.password: <MongoDB user password>
    alternative.mongo.port: 27017

To use the mongo returner, append '--return mongo' to the salt command.

.. code-block:: bash

    salt '*' test.ping --return mongo_return

To use the alternative configuration, append '--return_config alternative' to the salt command.

.. versionadded:: 2015.5.0

.. code-block:: bash

    salt '*' test.ping --return mongo_return --return_config alternative

To override individual configuration items, append --return_kwargs '{"key:": "value"}' to the salt command.

.. versionadded:: 2016.3.0

.. code-block:: bash

    salt '*' test.ping --return mongo --return_kwargs '{"db": "another-salt"}'

To override individual configuration items, append --return_kwargs '{"key:": "value"}' to the salt command.

.. versionadded:: 2016.3.0

.. code-block:: bash

    salt '*' test.ping --return mongo --return_kwargs '{"db": "another-salt"}'

"""
import logging
import salt.returners
import salt.utils.jid
from salt.utils.versions import Version
try:
    import pymongo
    PYMONGO_VERSION = Version(pymongo.version)
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False
log = logging.getLogger(__name__)
__virtualname__ = 'mongo'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    if not HAS_PYMONGO:
        return (False, 'Could not import mongo returner; pymongo is not installed.')
    return 'mongo_return'

def _remove_dots(src):
    if False:
        i = 10
        return i + 15
    '\n    Remove dots from the given data structure\n    '
    output = {}
    for (key, val) in src.items():
        if isinstance(val, dict):
            val = _remove_dots(val)
        output[key.replace('.', '-')] = val
    return output

def _get_options(ret):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the monogo_return options from salt.\n    '
    attrs = {'host': 'host', 'port': 'port', 'db': 'db', 'user': 'user', 'password': 'password', 'indexes': 'indexes'}
    _options = salt.returners.get_returner_options(__virtualname__, ret, attrs, __salt__=__salt__, __opts__=__opts__)
    return _options

def _get_conn(ret):
    if False:
        i = 10
        return i + 15
    '\n    Return a mongodb connection object\n    '
    _options = _get_options(ret)
    host = _options.get('host')
    port = _options.get('port')
    db_ = _options.get('db')
    user = _options.get('user')
    password = _options.get('password')
    indexes = _options.get('indexes', False)
    if PYMONGO_VERSION > Version('2.3'):
        conn = pymongo.MongoClient(host, port)
    else:
        conn = pymongo.Connection(host, port)
    mdb = conn[db_]
    if user and password:
        mdb.authenticate(user, password)
    if indexes:
        if PYMONGO_VERSION > Version('2.3'):
            mdb.saltReturns.create_index('minion')
            mdb.saltReturns.create_index('jid')
            mdb.jobs.create_index('jid')
        else:
            mdb.saltReturns.ensure_index('minion')
            mdb.saltReturns.ensure_index('jid')
            mdb.jobs.ensure_index('jid')
    return (conn, mdb)

def returner(ret):
    if False:
        print('Hello World!')
    '\n    Return data to a mongodb server\n    '
    (conn, mdb) = _get_conn(ret)
    col = mdb[ret['id']]
    if isinstance(ret['return'], dict):
        back = _remove_dots(ret['return'])
    else:
        back = ret['return']
    if isinstance(ret, dict):
        full_ret = _remove_dots(ret)
    else:
        full_ret = ret
    log.debug(back)
    sdata = {'minion': ret['id'], 'jid': ret['jid'], 'return': back, 'fun': ret['fun'], 'full_ret': full_ret}
    if 'out' in ret:
        sdata['out'] = ret['out']
    if PYMONGO_VERSION > Version('2.3'):
        mdb.saltReturns.insert_one(sdata.copy())
    else:
        mdb.saltReturns.insert(sdata.copy())

def get_jid(jid):
    if False:
        print('Hello World!')
    '\n    Return the return information associated with a jid\n    '
    (conn, mdb) = _get_conn(ret=None)
    ret = {}
    rdata = mdb.saltReturns.find({'jid': jid}, {'_id': 0})
    if rdata:
        for data in rdata:
            minion = data['minion']
            ret[minion] = data['full_ret']
    return ret

def get_fun(fun):
    if False:
        i = 10
        return i + 15
    '\n    Return the most recent jobs that have executed the named function\n    '
    (conn, mdb) = _get_conn(ret=None)
    ret = {}
    rdata = mdb.saltReturns.find_one({'fun': fun}, {'_id': 0})
    if rdata:
        ret = rdata
    return ret

def prep_jid(nocache=False, passed_jid=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do any work necessary to prepare a JID, including sending a custom id\n    '
    return passed_jid if passed_jid is not None else salt.utils.jid.gen_jid(__opts__)

def save_minions(jid, minions, syndic_id=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Included for API consistency\n    '