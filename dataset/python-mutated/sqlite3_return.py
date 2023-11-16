"""
Insert minion return data into a sqlite3 database

:maintainer:    Mickey Malone <mickey.malone@gmail.com>
:maturity:      New
:depends:       None
:platform:      All

Sqlite3 is a serverless database that lives in a single file.
In order to use this returner the database file must exist,
have the appropriate schema defined, and be accessible to the
user whom the minion process is running as. This returner
requires the following values configured in the master or
minion config:

.. code-block:: yaml

    sqlite3.database: /usr/lib/salt/salt.db
    sqlite3.timeout: 5.0

Alternative configuration values can be used by prefacing the configuration.
Any values not found in the alternative configuration will be pulled from
the default location:

.. code-block:: yaml

    alternative.sqlite3.database: /usr/lib/salt/salt.db
    alternative.sqlite3.timeout: 5.0

Use the commands to create the sqlite3 database and tables:

.. code-block:: sql

    sqlite3 /usr/lib/salt/salt.db << EOF
    --
    -- Table structure for table 'jids'
    --

    CREATE TABLE jids (
      jid TEXT PRIMARY KEY,
      load TEXT NOT NULL
      );

    --
    -- Table structure for table 'salt_returns'
    --

    CREATE TABLE salt_returns (
      fun TEXT KEY,
      jid TEXT KEY,
      id TEXT KEY,
      fun_args TEXT,
      date TEXT NOT NULL,
      full_ret TEXT NOT NULL,
      success TEXT NOT NULL
      );
    EOF

To use the sqlite returner, append '--return sqlite3' to the salt command.

.. code-block:: bash

    salt '*' test.ping --return sqlite3

To use the alternative configuration, append '--return_config alternative' to the salt command.

.. versionadded:: 2015.5.0

.. code-block:: bash

    salt '*' test.ping --return sqlite3 --return_config alternative

To override individual configuration items, append --return_kwargs '{"key:": "value"}' to the salt command.

.. versionadded:: 2016.3.0

.. code-block:: bash

    salt '*' test.ping --return sqlite3 --return_kwargs '{"db": "/var/lib/salt/another-salt.db"}'

"""
import datetime
import logging
import salt.returners
import salt.utils.jid
import salt.utils.json
try:
    import sqlite3
    HAS_SQLITE3 = True
except ImportError:
    HAS_SQLITE3 = False
log = logging.getLogger(__name__)
__virtualname__ = 'sqlite3'

def __virtual__():
    if False:
        print('Hello World!')
    if not HAS_SQLITE3:
        return (False, 'Could not import sqlite3 returner; sqlite3 is not installed.')
    return __virtualname__

def _get_options(ret=None):
    if False:
        print('Hello World!')
    '\n    Get the SQLite3 options from salt.\n    '
    attrs = {'database': 'database', 'timeout': 'timeout'}
    _options = salt.returners.get_returner_options(__virtualname__, ret, attrs, __salt__=__salt__, __opts__=__opts__)
    return _options

def _get_conn(ret=None):
    if False:
        print('Hello World!')
    '\n    Return a sqlite3 database connection\n    '
    _options = _get_options(ret)
    database = _options.get('database')
    timeout = _options.get('timeout')
    if not database:
        raise Exception('sqlite3 config option "sqlite3.database" is missing')
    if not timeout:
        raise Exception('sqlite3 config option "sqlite3.timeout" is missing')
    log.debug('Connecting the sqlite3 database: %s timeout: %s', database, timeout)
    conn = sqlite3.connect(database, timeout=float(timeout))
    return conn

def _close_conn(conn):
    if False:
        print('Hello World!')
    '\n    Close the sqlite3 database connection\n    '
    log.debug('Closing the sqlite3 database connection')
    conn.commit()
    conn.close()

def returner(ret):
    if False:
        for i in range(10):
            print('nop')
    '\n    Insert minion return data into the sqlite3 database\n    '
    log.debug('sqlite3 returner <returner> called with data: %s', ret)
    conn = _get_conn(ret)
    cur = conn.cursor()
    sql = 'INSERT INTO salt_returns\n             (fun, jid, id, fun_args, date, full_ret, success)\n             VALUES (:fun, :jid, :id, :fun_args, :date, :full_ret, :success)'
    cur.execute(sql, {'fun': ret['fun'], 'jid': ret['jid'], 'id': ret['id'], 'fun_args': str(ret['fun_args']) if ret.get('fun_args') else None, 'date': str(datetime.datetime.now()), 'full_ret': salt.utils.json.dumps(ret['return']), 'success': ret.get('success', '')})
    _close_conn(conn)

def save_load(jid, load, minions=None):
    if False:
        print('Hello World!')
    '\n    Save the load to the specified jid\n    '
    log.debug('sqlite3 returner <save_load> called jid: %s load: %s', jid, load)
    conn = _get_conn(ret=None)
    cur = conn.cursor()
    sql = 'INSERT INTO jids (jid, load) VALUES (:jid, :load)'
    cur.execute(sql, {'jid': jid, 'load': salt.utils.json.dumps(load)})
    _close_conn(conn)

def save_minions(jid, minions, syndic_id=None):
    if False:
        return 10
    '\n    Included for API consistency\n    '

def get_load(jid):
    if False:
        while True:
            i = 10
    '\n    Return the load from a specified jid\n    '
    log.debug('sqlite3 returner <get_load> called jid: %s', jid)
    conn = _get_conn(ret=None)
    cur = conn.cursor()
    sql = 'SELECT load FROM jids WHERE jid = :jid'
    cur.execute(sql, {'jid': jid})
    data = cur.fetchone()
    if data:
        return salt.utils.json.loads(data[0].encode())
    _close_conn(conn)
    return {}

def get_jid(jid):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the information returned from a specified jid\n    '
    log.debug('sqlite3 returner <get_jid> called jid: %s', jid)
    conn = _get_conn(ret=None)
    cur = conn.cursor()
    sql = 'SELECT id, full_ret FROM salt_returns WHERE jid = :jid'
    cur.execute(sql, {'jid': jid})
    data = cur.fetchone()
    log.debug('query result: %s', data)
    ret = {}
    if data and len(data) > 1:
        ret = {str(data[0]): {'return': salt.utils.json.loads(data[1])}}
        log.debug('ret: %s', ret)
    _close_conn(conn)
    return ret

def get_fun(fun):
    if False:
        return 10
    '\n    Return a dict of the last function called for all minions\n    '
    log.debug('sqlite3 returner <get_fun> called fun: %s', fun)
    conn = _get_conn(ret=None)
    cur = conn.cursor()
    sql = 'SELECT s.id, s.full_ret, s.jid\n            FROM salt_returns s\n            JOIN ( SELECT MAX(jid) AS jid FROM salt_returns GROUP BY fun, id) max\n            ON s.jid = max.jid\n            WHERE s.fun = :fun\n            '
    cur.execute(sql, {'fun': fun})
    data = cur.fetchall()
    ret = {}
    if data:
        data.pop()
        for (minion, ret) in data:
            ret[minion] = salt.utils.json.loads(ret)
    _close_conn(conn)
    return ret

def get_jids():
    if False:
        i = 10
        return i + 15
    '\n    Return a list of all job ids\n    '
    log.debug('sqlite3 returner <get_jids> called')
    conn = _get_conn(ret=None)
    cur = conn.cursor()
    sql = 'SELECT jid, load FROM jids'
    cur.execute(sql)
    data = cur.fetchall()
    ret = {}
    for (jid, load) in data:
        ret[jid] = salt.utils.jid.format_jid_instance(jid, salt.utils.json.loads(load))
    _close_conn(conn)
    return ret

def get_minions():
    if False:
        print('Hello World!')
    '\n    Return a list of minions\n    '
    log.debug('sqlite3 returner <get_minions> called')
    conn = _get_conn(ret=None)
    cur = conn.cursor()
    sql = 'SELECT DISTINCT id FROM salt_returns'
    cur.execute(sql)
    data = cur.fetchall()
    ret = []
    for minion in data:
        ret.append(minion[0])
    _close_conn(conn)
    return ret

def prep_jid(nocache=False, passed_jid=None):
    if False:
        print('Hello World!')
    '\n    Do any work necessary to prepare a JID, including sending a custom id\n    '
    return passed_jid if passed_jid is not None else salt.utils.jid.gen_jid(__opts__)