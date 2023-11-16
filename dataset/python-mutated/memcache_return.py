"""
Return data to a memcache server

To enable this returner the minion will need the python client for memcache
installed and the following values configured in the minion or master
config, these are the defaults.

.. code-block:: yaml

    memcache.host: 'localhost'
    memcache.port: '11211'

Alternative configuration values can be used by prefacing the configuration.
Any values not found in the alternative configuration will be pulled from
the default location.

.. code-block:: yaml

    alternative.memcache.host: 'localhost'
    alternative.memcache.port: '11211'

python2-memcache uses 'localhost' and '11211' as syntax on connection.

To use the memcache returner, append '--return memcache' to the salt command.

.. code-block:: bash

    salt '*' test.ping --return memcache

To use the alternative configuration, append '--return_config alternative' to the salt command.

.. versionadded:: 2015.5.0

.. code-block:: bash

    salt '*' test.ping --return memcache --return_config alternative

To override individual configuration items, append --return_kwargs '{"key:": "value"}' to the salt command.

.. versionadded:: 2016.3.0

.. code-block:: bash

    salt '*' test.ping --return memcache --return_kwargs '{"host": "hostname.domain.com"}'

"""
import logging
import salt.returners
import salt.utils.jid
import salt.utils.json
log = logging.getLogger(__name__)
try:
    import memcache
    HAS_MEMCACHE = True
except ImportError:
    HAS_MEMCACHE = False
__virtualname__ = 'memcache'

def __virtual__():
    if False:
        return 10
    if not HAS_MEMCACHE:
        return (False, 'Could not import memcache returner; memcache python client is not installed.')
    return __virtualname__

def _get_options(ret=None):
    if False:
        return 10
    '\n    Get the memcache options from salt.\n    '
    attrs = {'host': 'host', 'port': 'port'}
    _options = salt.returners.get_returner_options(__virtualname__, ret, attrs, __salt__=__salt__, __opts__=__opts__)
    return _options

def _get_serv(ret):
    if False:
        i = 10
        return i + 15
    '\n    Return a memcache server object\n    '
    _options = _get_options(ret)
    host = _options.get('host')
    port = _options.get('port')
    log.debug('memcache server: %s:%s', host, port)
    if not host or not port:
        log.error('Host or port not defined in salt config')
        return
    memcacheoptions = (host, port)
    return memcache.Client(['{}:{}'.format(*memcacheoptions)], debug=0)

def _get_list(serv, key):
    if False:
        while True:
            i = 10
    value = serv.get(key)
    if value:
        return value.strip(',').split(',')
    return []

def _append_list(serv, key, value):
    if False:
        i = 10
        return i + 15
    if value in _get_list(serv, key):
        return
    r = serv.append(key, '{},'.format(value))
    if not r:
        serv.add(key, '{},'.format(value))

def prep_jid(nocache=False, passed_jid=None):
    if False:
        return 10
    '\n    Do any work necessary to prepare a JID, including sending a custom id\n    '
    return passed_jid if passed_jid is not None else salt.utils.jid.gen_jid(__opts__)

def returner(ret):
    if False:
        return 10
    '\n    Return data to a memcache data store\n    '
    serv = _get_serv(ret)
    minion = ret['id']
    jid = ret['jid']
    fun = ret['fun']
    rets = salt.utils.json.dumps(ret)
    serv.set('{}:{}'.format(jid, minion), rets)
    serv.set('{}:{}'.format(fun, minion), rets)
    _append_list(serv, 'minions', minion)
    _append_list(serv, 'jids', jid)

def save_load(jid, load, minions=None):
    if False:
        i = 10
        return i + 15
    '\n    Save the load to the specified jid\n    '
    serv = _get_serv(ret=None)
    serv.set(jid, salt.utils.json.dumps(load))
    _append_list(serv, 'jids', jid)

def save_minions(jid, minions, syndic_id=None):
    if False:
        i = 10
        return i + 15
    '\n    Included for API consistency\n    '

def get_load(jid):
    if False:
        i = 10
        return i + 15
    '\n    Return the load data that marks a specified jid\n    '
    serv = _get_serv(ret=None)
    data = serv.get(jid)
    if data:
        return salt.utils.json.loads(data)
    return {}

def get_jid(jid):
    if False:
        return 10
    '\n    Return the information returned when the specified job id was executed\n    '
    serv = _get_serv(ret=None)
    minions = _get_list(serv, 'minions')
    returns = serv.get_multi(minions, key_prefix='{}:'.format(jid))
    ret = {}
    for (minion, data) in returns.items():
        ret[minion] = salt.utils.json.loads(data)
    return ret

def get_fun(fun):
    if False:
        print('Hello World!')
    '\n    Return a dict of the last function called for all minions\n    '
    serv = _get_serv(ret=None)
    minions = _get_list(serv, 'minions')
    returns = serv.get_multi(minions, key_prefix='{}:'.format(fun))
    ret = {}
    for (minion, data) in returns.items():
        ret[minion] = salt.utils.json.loads(data)
    return ret

def get_jids():
    if False:
        while True:
            i = 10
    '\n    Return a list of all job ids\n    '
    serv = _get_serv(ret=None)
    jids = _get_list(serv, 'jids')
    loads = serv.get_multi(jids)
    ret = {}
    for (jid, load) in loads.items():
        ret[jid] = salt.utils.jid.format_jid_instance(jid, salt.utils.json.loads(load))
    return ret

def get_minions():
    if False:
        i = 10
        return i + 15
    '\n    Return a list of minions\n    '
    serv = _get_serv(ret=None)
    return _get_list(serv, 'minions')