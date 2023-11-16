"""
InfluxDB - A distributed time series database

Module to provide InfluxDB compatibility to Salt (compatible with InfluxDB
version 0.9+)

:depends:    - influxdb Python module (>= 3.0.0)

:configuration: This module accepts connection configuration details either as
    parameters or as configuration settings in /etc/salt/minion on the relevant
    minions::

        influxdb.host: 'localhost'
        influxdb.port: 8086
        influxdb.user: 'root'
        influxdb.password: 'root'

    This data can also be passed into pillar. Options passed into opts will
    overwrite options passed into pillar.

    Most functions in this module allow you to override or provide some or all
    of these settings via keyword arguments::

        salt '*' influxdb.foo_function influxdb_user='influxadmin' influxdb_password='s3cr1t'

    would override ``user`` and ``password`` while still using the defaults for
    ``host`` and ``port``.
"""
import collections
import logging
from collections.abc import Sequence
import salt.utils.json
from salt.state import STATE_INTERNAL_KEYWORDS as _STATE_INTERNAL_KEYWORDS
try:
    import influxdb
    HAS_INFLUXDB = True
except ImportError:
    HAS_INFLUXDB = False
log = logging.getLogger(__name__)
__virtualname__ = 'influxdb'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if influxdb lib is present\n    '
    if HAS_INFLUXDB:
        return __virtualname__
    return (False, 'The influxdb execution module could not be loaded:influxdb library not available.')

def _client(influxdb_user=None, influxdb_password=None, influxdb_host=None, influxdb_port=None, **client_args):
    if False:
        return 10
    if not influxdb_user:
        influxdb_user = __salt__['config.option']('influxdb.user', 'root')
    if not influxdb_password:
        influxdb_password = __salt__['config.option']('influxdb.password', 'root')
    if not influxdb_host:
        influxdb_host = __salt__['config.option']('influxdb.host', 'localhost')
    if not influxdb_port:
        influxdb_port = __salt__['config.option']('influxdb.port', 8086)
    for ignore in _STATE_INTERNAL_KEYWORDS:
        if ignore in client_args:
            del client_args[ignore]
    return influxdb.InfluxDBClient(host=influxdb_host, port=influxdb_port, username=influxdb_user, password=influxdb_password, **client_args)

def list_dbs(**client_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all InfluxDB databases.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.list_dbs\n    "
    client = _client(**client_args)
    return client.get_list_database()

def db_exists(name, **client_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Checks if a database exists in InfluxDB.\n\n    name\n        Name of the database to check.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.db_exists <name>\n    "
    if name in [db['name'] for db in list_dbs(**client_args)]:
        return True
    return False

def create_db(name, **client_args):
    if False:
        while True:
            i = 10
    "\n    Create a database.\n\n    name\n        Name of the database to create.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.create_db <name>\n    "
    if db_exists(name, **client_args):
        log.info("DB '%s' already exists", name)
        return False
    client = _client(**client_args)
    client.create_database(name)
    return True

def drop_db(name, **client_args):
    if False:
        i = 10
        return i + 15
    "\n    Drop a database.\n\n    name\n        Name of the database to drop.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.drop_db <name>\n    "
    if not db_exists(name, **client_args):
        log.info("DB '%s' does not exist", name)
        return False
    client = _client(**client_args)
    client.drop_database(name)
    return True

def list_users(**client_args):
    if False:
        return 10
    "\n    List all users.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.list_users\n    "
    client = _client(**client_args)
    return client.get_list_users()

def user_exists(name, **client_args):
    if False:
        i = 10
        return i + 15
    "\n    Check if a user exists.\n\n    name\n        Name of the user to check.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.user_exists <name>\n    "
    if user_info(name, **client_args):
        return True
    return False

def user_info(name, **client_args):
    if False:
        i = 10
        return i + 15
    "\n    Get information about given user.\n\n    name\n        Name of the user for which to get information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.user_info <name>\n    "
    matching_users = (user for user in list_users(**client_args) if user.get('user') == name)
    try:
        return next(matching_users)
    except StopIteration:
        pass

def create_user(name, passwd, admin=False, **client_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a user.\n\n    name\n        Name of the user to create.\n\n    passwd\n        Password of the new user.\n\n    admin : False\n        Whether the user should have cluster administration\n        privileges or not.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.create_user <name> <password>\n        salt '*' influxdb.create_user <name> <password> admin=True\n    "
    if user_exists(name, **client_args):
        log.info("User '%s' already exists", name)
        return False
    client = _client(**client_args)
    client.create_user(name, passwd, admin)
    return True

def set_user_password(name, passwd, **client_args):
    if False:
        return 10
    "\n    Change password of a user.\n\n    name\n        Name of the user for whom to set the password.\n\n    passwd\n        New password of the user.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.set_user_password <name> <password>\n    "
    if not user_exists(name, **client_args):
        log.info("User '%s' does not exist", name)
        return False
    client = _client(**client_args)
    client.set_user_password(name, passwd)
    return True

def grant_admin_privileges(name, **client_args):
    if False:
        return 10
    "\n    Grant cluster administration privileges to a user.\n\n    name\n        Name of the user to whom admin privileges will be granted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.grant_admin_privileges <name>\n    "
    client = _client(**client_args)
    client.grant_admin_privileges(name)
    return True

def revoke_admin_privileges(name, **client_args):
    if False:
        print('Hello World!')
    "\n    Revoke cluster administration privileges from a user.\n\n    name\n        Name of the user from whom admin privileges will be revoked.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.revoke_admin_privileges <name>\n    "
    client = _client(**client_args)
    client.revoke_admin_privileges(name)
    return True

def remove_user(name, **client_args):
    if False:
        i = 10
        return i + 15
    "\n    Remove a user.\n\n    name\n        Name of the user to remove\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.remove_user <name>\n    "
    if not user_exists(name, **client_args):
        log.info("User '%s' does not exist", name)
        return False
    client = _client(**client_args)
    client.drop_user(name)
    return True

def get_retention_policy(database, name, **client_args):
    if False:
        return 10
    "\n    Get an existing retention policy.\n\n    database\n        Name of the database for which the retention policy was\n        defined.\n\n    name\n        Name of the retention policy.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.get_retention_policy metrics default\n    "
    client = _client(**client_args)
    try:
        return next((p for p in client.get_list_retention_policies(database) if p.get('name') == name))
    except StopIteration:
        return {}

def retention_policy_exists(database, name, **client_args):
    if False:
        print('Hello World!')
    "\n    Check if retention policy with given name exists.\n\n    database\n        Name of the database for which the retention policy was\n        defined.\n\n    name\n        Name of the retention policy to check.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.retention_policy_exists metrics default\n    "
    if get_retention_policy(database, name, **client_args):
        return True
    return False

def drop_retention_policy(database, name, **client_args):
    if False:
        return 10
    "\n    Drop a retention policy.\n\n    database\n        Name of the database for which the retention policy will be dropped.\n\n    name\n        Name of the retention policy to drop.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.drop_retention_policy mydb mypr\n    "
    client = _client(**client_args)
    client.drop_retention_policy(name, database)
    return True

def create_retention_policy(database, name, duration, replication, default=False, **client_args):
    if False:
        while True:
            i = 10
    "\n    Create a retention policy.\n\n    database\n        Name of the database for which the retention policy will be created.\n\n    name\n        Name of the new retention policy.\n\n    duration\n        Duration of the new retention policy.\n\n        Durations such as 1h, 90m, 12h, 7d, and 4w, are all supported and mean\n        1 hour, 90 minutes, 12 hours, 7 day, and 4 weeks, respectively. For\n        infinite retention – meaning the data will never be deleted – use 'INF'\n        for duration. The minimum retention period is 1 hour.\n\n    replication\n        Replication factor of the retention policy.\n\n        This determines how many independent copies of each data point are\n        stored in a cluster.\n\n    default : False\n        Whether or not the policy as default will be set as default.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.create_retention_policy metrics default 1d 1\n    "
    client = _client(**client_args)
    client.create_retention_policy(name, duration, replication, database, default)
    return True

def alter_retention_policy(database, name, duration, replication, default=False, **client_args):
    if False:
        while True:
            i = 10
    "\n    Modify an existing retention policy.\n\n    name\n        Name of the retention policy to modify.\n\n    database\n        Name of the database for which the retention policy was defined.\n\n    duration\n        New duration of given retention policy.\n\n        Durations such as 1h, 90m, 12h, 7d, and 4w, are all supported\n        and mean 1 hour, 90 minutes, 12 hours, 7 day, and 4 weeks,\n        respectively. For infinite retention – meaning the data will\n        never be deleted – use 'INF' for duration.\n        The minimum retention period is 1 hour.\n\n    replication\n        New replication of given retention policy.\n\n        This determines how many independent copies of each data point are\n        stored in a cluster.\n\n    default : False\n        Whether or not to set the modified policy as default.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.alter_retention_policy metrics default 1d 1\n    "
    client = _client(**client_args)
    client.alter_retention_policy(name, database, duration, replication, default)
    return True

def list_privileges(name, **client_args):
    if False:
        print('Hello World!')
    "\n    List privileges from a user.\n\n    name\n        Name of the user from whom privileges will be listed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.list_privileges <name>\n    "
    client = _client(**client_args)
    res = {}
    for item in client.get_list_privileges(name):
        res[item['database']] = item['privilege'].split()[0].lower()
    return res

def grant_privilege(database, privilege, username, **client_args):
    if False:
        while True:
            i = 10
    "\n    Grant a privilege on a database to a user.\n\n    database\n        Name of the database to grant the privilege on.\n\n    privilege\n        Privilege to grant. Can be one of 'read', 'write' or 'all'.\n\n    username\n        Name of the user to grant the privilege to.\n    "
    client = _client(**client_args)
    client.grant_privilege(privilege, database, username)
    return True

def revoke_privilege(database, privilege, username, **client_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Revoke a privilege on a database from a user.\n\n    database\n        Name of the database to grant the privilege on.\n\n    privilege\n        Privilege to grant. Can be one of 'read', 'write' or 'all'.\n\n    username\n        Name of the user to grant the privilege to.\n    "
    client = _client(**client_args)
    client.revoke_privilege(privilege, database, username)
    return True

def continuous_query_exists(database, name, **client_args):
    if False:
        return 10
    "\n    Check if continuous query with given name exists on the database.\n\n    database\n        Name of the database for which the continuous query was\n        defined.\n\n    name\n        Name of the continuous query to check.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.continuous_query_exists metrics default\n    "
    if get_continuous_query(database, name, **client_args):
        return True
    return False

def get_continuous_query(database, name, **client_args):
    if False:
        i = 10
        return i + 15
    "\n    Get an existing continuous query.\n\n    database\n        Name of the database for which the continuous query was\n        defined.\n\n    name\n        Name of the continuous query to get.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.get_continuous_query mydb cq_month\n    "
    client = _client(**client_args)
    try:
        for (db, cqs) in client.query('SHOW CONTINUOUS QUERIES').items():
            if db[0] == database:
                return next((cq for cq in cqs if cq.get('name') == name))
    except StopIteration:
        return {}
    return {}

def create_continuous_query(database, name, query, resample_time=None, coverage_period=None, **client_args):
    if False:
        return 10
    "\n    Create a continuous query.\n\n    database\n        Name of the database for which the continuous query will be\n        created on.\n\n    name\n        Name of the continuous query to create.\n\n    query\n        The continuous query string.\n\n    resample_time : None\n        Duration between continuous query resampling.\n\n    coverage_period : None\n        Duration specifying time period per sample.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.create_continuous_query mydb cq_month 'SELECT mean(*) INTO mydb.a_month.:MEASUREMENT FROM mydb.a_week./.*/ GROUP BY time(5m), *'"
    client = _client(**client_args)
    full_query = 'CREATE CONTINUOUS QUERY {name} ON {database}'
    if resample_time:
        full_query += ' RESAMPLE EVERY {resample_time}'
    if coverage_period:
        full_query += ' FOR {coverage_period}'
    full_query += ' BEGIN {query} END'
    query = full_query.format(name=name, database=database, query=query, resample_time=resample_time, coverage_period=coverage_period)
    client.query(query)
    return True

def drop_continuous_query(database, name, **client_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Drop a continuous query.\n\n    database\n        Name of the database for which the continuous query will\n        be drop from.\n\n    name\n        Name of the continuous query to drop.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.drop_continuous_query mydb my_cq\n    "
    client = _client(**client_args)
    query = 'DROP CONTINUOUS QUERY {} ON {}'.format(name, database)
    client.query(query)
    return True

def _pull_query_results(resultset):
    if False:
        print('Hello World!')
    '\n    Parses a ResultSet returned from InfluxDB into a dictionary of results,\n    grouped by series names and optional JSON-encoded grouping tags.\n    '
    _results = collections.defaultdict(lambda : {})
    for (_header, _values) in resultset.items():
        (_header, _group_tags) = _header
        if _group_tags:
            _results[_header][salt.utils.json.dumps(_group_tags)] = [_value for _value in _values]
        else:
            _results[_header] = [_value for _value in _values]
    return dict(sorted(_results.items()))

def query(database, query, **client_args):
    if False:
        return 10
    '\n    Execute a query.\n\n    database\n        Name of the database to query on.\n\n    query\n        InfluxQL query string.\n    '
    client = _client(**client_args)
    _result = client.query(query, database=database)
    if isinstance(_result, Sequence):
        return [_pull_query_results(_query_result) for _query_result in _result if _query_result]
    return [_pull_query_results(_result) if _result else {}]