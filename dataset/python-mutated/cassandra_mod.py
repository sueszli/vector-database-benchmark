"""

.. warning::

    The `cassandra` module is deprecated in favor of the `cassandra_cql`
    module.

Cassandra NoSQL Database Module

:depends:   - pycassa Cassandra Python adapter
:configuration:
    The location of the 'nodetool' command, host, and thrift port needs to be
    specified via pillar::

        cassandra.nodetool: /usr/local/bin/nodetool
        cassandra.host: localhost
        cassandra.thrift_port: 9160
"""
import logging
import salt.utils.path
from salt.utils.versions import warn_until_date
log = logging.getLogger(__name__)
HAS_PYCASSA = False
try:
    from pycassa.system_manager import SystemManager
    HAS_PYCASSA = True
except ImportError:
    pass

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if pycassa is available and the system is configured\n    '
    if not HAS_PYCASSA:
        return (False, 'The cassandra execution module cannot be loaded: pycassa not installed.')
    warn_until_date('20240101', 'The cassandra returner is broken and deprecated, and will be removed after {date}. Use the cassandra_cql returner instead')
    if HAS_PYCASSA and salt.utils.path.which('nodetool'):
        return 'cassandra'
    return (False, 'The cassandra execution module cannot be loaded: nodetool not found.')

def _nodetool(cmd):
    if False:
        i = 10
        return i + 15
    '\n    Internal cassandra nodetool wrapper. Some functions are not\n    available via pycassa so we must rely on nodetool.\n    '
    nodetool = __salt__['config.option']('cassandra.nodetool')
    host = __salt__['config.option']('cassandra.host')
    return __salt__['cmd.run_stdout']('{} -h {} {}'.format(nodetool, host, cmd))

def _sys_mgr():
    if False:
        while True:
            i = 10
    '\n    Return a pycassa system manager connection object\n    '
    thrift_port = str(__salt__['config.option']('cassandra.THRIFT_PORT'))
    host = __salt__['config.option']('cassandra.host')
    return SystemManager('{}:{}'.format(host, thrift_port))

def compactionstats():
    if False:
        while True:
            i = 10
    "\n    Return compactionstats info\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.compactionstats\n    "
    return _nodetool('compactionstats')

def version():
    if False:
        print('Hello World!')
    "\n    Return the cassandra version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.version\n    "
    return _nodetool('version')

def netstats():
    if False:
        i = 10
        return i + 15
    "\n    Return netstats info\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.netstats\n    "
    return _nodetool('netstats')

def tpstats():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return tpstats info\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.tpstats\n    "
    return _nodetool('tpstats')

def info():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return cassandra node info\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.info\n    "
    return _nodetool('info')

def ring():
    if False:
        print('Hello World!')
    "\n    Return cassandra ring info\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.ring\n    "
    return _nodetool('ring')

def keyspaces():
    if False:
        i = 10
        return i + 15
    "\n    Return existing keyspaces\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.keyspaces\n    "
    sys = _sys_mgr()
    return sys.list_keyspaces()

def column_families(keyspace=None):
    if False:
        i = 10
        return i + 15
    "\n    Return existing column families for all keyspaces\n    or just the provided one.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.column_families\n        salt '*' cassandra.column_families <keyspace>\n    "
    sys = _sys_mgr()
    ksps = sys.list_keyspaces()
    if keyspace:
        if keyspace in ksps:
            return list(sys.get_keyspace_column_families(keyspace).keys())
        else:
            return None
    else:
        ret = {}
        for kspace in ksps:
            ret[kspace] = list(sys.get_keyspace_column_families(kspace).keys())
        return ret

def column_family_definition(keyspace, column_family):
    if False:
        print('Hello World!')
    "\n    Return a dictionary of column family definitions for the given\n    keyspace/column_family\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cassandra.column_family_definition <keyspace> <column_family>\n\n    "
    sys = _sys_mgr()
    try:
        return vars(sys.get_keyspace_column_families(keyspace)[column_family])
    except Exception:
        log.debug('Invalid Keyspace/CF combination')
        return None