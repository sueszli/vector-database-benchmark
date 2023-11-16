"""
Return data to a cassandra server

.. versionadded:: 2015.5.0

:maintainer:    Corin Kochenower<ckochenower@saltstack.com>
:maturity:      new as of 2015.2
:depends:       salt.modules.cassandra_cql
:depends:       DataStax Python Driver for Apache Cassandra
                https://github.com/datastax/python-driver
                pip install cassandra-driver
:platform:      all

:configuration:
    To enable this returner, the minion will need the DataStax Python Driver
    for Apache Cassandra ( https://github.com/datastax/python-driver )
    installed and the following values configured in the minion or master
    config. The list of cluster IPs must include at least one cassandra node
    IP address. No assumption or default will be used for the cluster IPs.
    The cluster IPs will be tried in the order listed. The port, username,
    and password values shown below will be the assumed defaults if you do
    not provide values.:

    .. code-block:: yaml

        cassandra:
          cluster:
            - 192.168.50.11
            - 192.168.50.12
            - 192.168.50.13
          port: 9042
          username: salt
          password: salt

    Use the following cassandra database schema:

    .. code-block:: text

        CREATE KEYSPACE IF NOT EXISTS salt
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor' : 1};

        CREATE USER IF NOT EXISTS salt WITH PASSWORD 'salt' NOSUPERUSER;

        GRANT ALL ON KEYSPACE salt TO salt;

        USE salt;

        CREATE TABLE IF NOT EXISTS salt.salt_returns (
            jid text,
            minion_id text,
            fun text,
            alter_time timestamp,
            full_ret text,
            return text,
            success boolean,
            PRIMARY KEY (jid, minion_id, fun)
        ) WITH CLUSTERING ORDER BY (minion_id ASC, fun ASC);
        CREATE INDEX IF NOT EXISTS salt_returns_minion_id ON salt.salt_returns (minion_id);
        CREATE INDEX IF NOT EXISTS salt_returns_fun ON salt.salt_returns (fun);

        CREATE TABLE IF NOT EXISTS salt.jids (
            jid text PRIMARY KEY,
            load text
        );

        CREATE TABLE IF NOT EXISTS salt.minions (
            minion_id text PRIMARY KEY,
            last_fun text
        );
        CREATE INDEX IF NOT EXISTS minions_last_fun ON salt.minions (last_fun);

        CREATE TABLE IF NOT EXISTS salt.salt_events (
            id timeuuid,
            tag text,
            alter_time timestamp,
            data text,
            master_id text,
            PRIMARY KEY (id, tag)
        ) WITH CLUSTERING ORDER BY (tag ASC);
        CREATE INDEX tag ON salt.salt_events (tag);


Required python modules: cassandra-driver

To use the cassandra returner, append '--return cassandra_cql' to the salt command. ex:

.. code-block:: bash

    salt '*' test.ping --return_cql cassandra

Note: if your Cassandra instance has not been tuned much you may benefit from
altering some timeouts in `cassandra.yaml` like so:

.. code-block:: yaml

    # How long the coordinator should wait for read operations to complete
    read_request_timeout_in_ms: 5000
    # How long the coordinator should wait for seq or index scans to complete
    range_request_timeout_in_ms: 20000
    # How long the coordinator should wait for writes to complete
    write_request_timeout_in_ms: 20000
    # How long the coordinator should wait for counter writes to complete
    counter_write_request_timeout_in_ms: 10000
    # How long a coordinator should continue to retry a CAS operation
    # that contends with other proposals for the same row
    cas_contention_timeout_in_ms: 5000
    # How long the coordinator should wait for truncates to complete
    # (This can be much longer, because unless auto_snapshot is disabled
    # we need to flush first so we can snapshot before removing the data.)
    truncate_request_timeout_in_ms: 60000
    # The default timeout for other, miscellaneous operations
    request_timeout_in_ms: 20000

As always, your mileage may vary and your Cassandra cluster may have different
needs.  SaltStack has seen situations where these timeouts can resolve
some stacktraces that appear to come from the Datastax Python driver.

"""
import logging
import time
import uuid
import salt.exceptions
import salt.returners
import salt.utils.jid
import salt.utils.json
from salt.exceptions import CommandExecutionError
try:
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.cluster import Cluster, NoHostAvailable
    from cassandra.connection import ConnectionException, ConnectionShutdown
    from cassandra.query import dict_factory
    HAS_CASSANDRA_DRIVER = True
except ImportError as e:
    HAS_CASSANDRA_DRIVER = False
log = logging.getLogger(__name__)
__virtualname__ = 'cassandra_cql'

def __virtual__():
    if False:
        i = 10
        return i + 15
    if not HAS_CASSANDRA_DRIVER:
        return (False, 'Could not import cassandra_cql returner; cassandra-driver is not installed.')
    return True

def returner(ret):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return data to one of potentially many clustered cassandra nodes\n    '
    query = 'INSERT INTO salt.salt_returns (\n                 jid, minion_id, fun, alter_time, full_ret, return, success\n               ) VALUES (?, ?, ?, ?, ?, ?, ?)'
    statement_arguments = ['{}'.format(ret['jid']), '{}'.format(ret['id']), '{}'.format(ret['fun']), int(time.time() * 1000), salt.utils.json.dumps(ret).replace("'", "''"), salt.utils.json.dumps(ret['return']).replace("'", "''"), ret.get('success', False)]
    try:
        __salt__['cassandra_cql.cql_query_with_prepare'](query, 'returner_return', tuple(statement_arguments), asynchronous=True)
    except CommandExecutionError:
        log.critical('Could not insert into salt_returns with Cassandra returner.')
        raise
    except Exception as e:
        log.critical('Unexpected error while inserting into salt_returns: %s', e)
        raise
    query = 'INSERT INTO salt.minions (\n                 minion_id, last_fun\n               ) VALUES (?, ?)'
    statement_arguments = ['{}'.format(ret['id']), '{}'.format(ret['fun'])]
    try:
        __salt__['cassandra_cql.cql_query_with_prepare'](query, 'returner_minion', tuple(statement_arguments), asynchronous=True)
    except CommandExecutionError:
        log.critical('Could not store minion ID with Cassandra returner.')
        raise
    except Exception as e:
        log.critical('Unexpected error while inserting minion ID into the minions table: %s', e)
        raise

def event_return(events):
    if False:
        i = 10
        return i + 15
    "\n    Return event to one of potentially many clustered cassandra nodes\n\n    Requires that configuration be enabled via 'event_return'\n    option in master config.\n\n    Cassandra does not support an auto-increment feature due to the\n    highly inefficient nature of creating a monotonically increasing\n    number across all nodes in a distributed database. Each event\n    will be assigned a uuid by the connecting client.\n    "
    for event in events:
        tag = event.get('tag', '')
        data = event.get('data', '')
        query = 'INSERT INTO salt.salt_events (\n                     id, alter_time, data, master_id, tag\n                   ) VALUES (\n                     ?, ?, ?, ?, ?)\n                 '
        statement_arguments = [str(uuid.uuid1()), int(time.time() * 1000), salt.utils.json.dumps(data).replace("'", "''"), __opts__['id'], tag]
        try:
            __salt__['cassandra_cql.cql_query_with_prepare'](query, 'salt_events', statement_arguments, asynchronous=True)
        except CommandExecutionError:
            log.critical('Could not store events with Cassandra returner.')
            raise
        except Exception as e:
            log.critical('Unexpected error while inserting into salt_events: %s', e)
            raise

def save_load(jid, load, minions=None):
    if False:
        print('Hello World!')
    '\n    Save the load to the specified jid id\n    '
    query = 'INSERT INTO salt.jids (\n                 jid, load\n               ) VALUES (?, ?)'
    statement_arguments = [jid, salt.utils.json.dumps(load).replace("'", "''")]
    try:
        __salt__['cassandra_cql.cql_query_with_prepare'](query, 'save_load', statement_arguments, asynchronous=True)
    except CommandExecutionError:
        log.critical('Could not save load in jids table.')
        raise
    except Exception as e:
        log.critical('Unexpected error while inserting into jids: %s', e)
        raise

def save_minions(jid, minions, syndic_id=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Included for API consistency\n    '

def get_load(jid):
    if False:
        i = 10
        return i + 15
    '\n    Return the load data that marks a specified jid\n    '
    query = 'SELECT load FROM salt.jids WHERE jid = ?;'
    ret = {}
    try:
        data = __salt__['cassandra_cql.cql_query_with_prepare'](query, 'get_load', [jid])
        if data:
            load = data[0].get('load')
            if load:
                ret = salt.utils.json.loads(load)
    except CommandExecutionError:
        log.critical('Could not get load from jids table.')
        raise
    except Exception as e:
        log.critical('Unexpected error while getting load from jids: %s', e)
        raise
    return ret

def get_jid(jid):
    if False:
        print('Hello World!')
    '\n    Return the information returned when the specified job id was executed\n    '
    query = 'SELECT minion_id, full_ret FROM salt.salt_returns WHERE jid = ?;'
    ret = {}
    try:
        data = __salt__['cassandra_cql.cql_query_with_prepare'](query, 'get_jid', [jid])
        if data:
            for row in data:
                minion = row.get('minion_id')
                full_ret = row.get('full_ret')
                if minion and full_ret:
                    ret[minion] = salt.utils.json.loads(full_ret)
    except CommandExecutionError:
        log.critical('Could not select job specific information.')
        raise
    except Exception as e:
        log.critical('Unexpected error while getting job specific information: %s', e)
        raise
    return ret

def get_fun(fun):
    if False:
        while True:
            i = 10
    '\n    Return a dict of the last function called for all minions\n    '
    query = 'SELECT minion_id, last_fun FROM salt.minions where last_fun = ?;'
    ret = {}
    try:
        data = __salt__['cassandra_cql.cql_query'](query, 'get_fun', [fun])
        if data:
            for row in data:
                minion = row.get('minion_id')
                last_fun = row.get('last_fun')
                if minion and last_fun:
                    ret[minion] = last_fun
    except CommandExecutionError:
        log.critical('Could not get the list of minions.')
        raise
    except Exception as e:
        log.critical('Unexpected error while getting list of minions: %s', e)
        raise
    return ret

def get_jids():
    if False:
        return 10
    '\n    Return a list of all job ids\n    '
    query = 'SELECT jid, load FROM salt.jids;'
    ret = {}
    try:
        data = __salt__['cassandra_cql.cql_query'](query)
        if data:
            for row in data:
                jid = row.get('jid')
                load = row.get('load')
                if jid and load:
                    ret[jid] = salt.utils.jid.format_jid_instance(jid, salt.utils.json.loads(load))
    except CommandExecutionError:
        log.critical('Could not get a list of all job ids.')
        raise
    except Exception as e:
        log.critical('Unexpected error while getting list of all job ids: %s', e)
        raise
    return ret

def get_minions():
    if False:
        while True:
            i = 10
    '\n    Return a list of minions\n    '
    query = 'SELECT DISTINCT minion_id FROM salt.minions;'
    ret = []
    try:
        data = __salt__['cassandra_cql.cql_query'](query)
        if data:
            for row in data:
                minion = row.get('minion_id')
                if minion:
                    ret.append(minion)
    except CommandExecutionError:
        log.critical('Could not get the list of minions.')
        raise
    except Exception as e:
        log.critical('Unexpected error while getting list of minions: %s', e)
        raise
    return ret

def prep_jid(nocache, passed_jid=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do any work necessary to prepare a JID, including sending a custom id\n    '
    return passed_jid if passed_jid is not None else salt.utils.jid.gen_jid(__opts__)