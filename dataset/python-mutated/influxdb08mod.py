"""
InfluxDB - A distributed time series database

Module to provide InfluxDB compatibility to Salt (compatible with InfluxDB
version 0.5-0.8)

.. versionadded:: 2014.7.0

:depends:    - influxdb Python module (>= 1.0.0)

:configuration: This module accepts connection configuration details either as
    parameters or as configuration settings in /etc/salt/minion on the relevant
    minions::

        influxdb08.host: 'localhost'
        influxdb08.port: 8086
        influxdb08.user: 'root'
        influxdb08.password: 'root'

    This data can also be passed into pillar. Options passed into opts will
    overwrite options passed into pillar.
"""
import logging
try:
    import influxdb.influxdb08
    HAS_INFLUXDB_08 = True
except ImportError:
    HAS_INFLUXDB_08 = False
log = logging.getLogger(__name__)
__virtualname__ = 'influxdb08'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if influxdb lib is present\n    '
    if HAS_INFLUXDB_08:
        return __virtualname__
    return (False, 'The influx execution module cannot be loaded: influxdb library not available.')

def _client(user=None, password=None, host=None, port=None):
    if False:
        return 10
    if not user:
        user = __salt__['config.option']('influxdb08.user', 'root')
    if not password:
        password = __salt__['config.option']('influxdb08.password', 'root')
    if not host:
        host = __salt__['config.option']('influxdb08.host', 'localhost')
    if not port:
        port = __salt__['config.option']('influxdb08.port', 8086)
    return influxdb.influxdb08.InfluxDBClient(host=host, port=port, username=user, password=password)

def db_list(user=None, password=None, host=None, port=None):
    if False:
        i = 10
        return i + 15
    "\n    List all InfluxDB databases\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.db_list\n        salt '*' influxdb08.db_list <user> <password> <host> <port>\n\n    "
    client = _client(user=user, password=password, host=host, port=port)
    return client.get_list_database()

def db_exists(name, user=None, password=None, host=None, port=None):
    if False:
        i = 10
        return i + 15
    "\n    Checks if a database exists in Influxdb\n\n    name\n        Database name to create\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.db_exists <name>\n        salt '*' influxdb08.db_exists <name> <user> <password> <host> <port>\n    "
    dbs = db_list(user, password, host, port)
    if not isinstance(dbs, list):
        return False
    return name in [db['name'] for db in dbs]

def db_create(name, user=None, password=None, host=None, port=None):
    if False:
        return 10
    "\n    Create a database\n\n    name\n        Database name to create\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.db_create <name>\n        salt '*' influxdb08.db_create <name> <user> <password> <host> <port>\n    "
    if db_exists(name, user, password, host, port):
        log.info("DB '%s' already exists", name)
        return False
    client = _client(user=user, password=password, host=host, port=port)
    client.create_database(name)
    return True

def db_remove(name, user=None, password=None, host=None, port=None):
    if False:
        print('Hello World!')
    "\n    Remove a database\n\n    name\n        Database name to remove\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.db_remove <name>\n        salt '*' influxdb08.db_remove <name> <user> <password> <host> <port>\n    "
    if not db_exists(name, user, password, host, port):
        log.info("DB '%s' does not exist", name)
        return False
    client = _client(user=user, password=password, host=host, port=port)
    return client.delete_database(name)

def user_list(database=None, user=None, password=None, host=None, port=None):
    if False:
        print('Hello World!')
    "\n    List cluster admins or database users.\n\n    If a database is specified: it will return database users list.\n    If a database is not specified: it will return cluster admins list.\n\n    database\n        The database to list the users from\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.user_list\n        salt '*' influxdb08.user_list <database>\n        salt '*' influxdb08.user_list <database> <user> <password> <host> <port>\n    "
    client = _client(user=user, password=password, host=host, port=port)
    if not database:
        return client.get_list_cluster_admins()
    client.switch_database(database)
    return client.get_list_users()

def user_exists(name, database=None, user=None, password=None, host=None, port=None):
    if False:
        return 10
    "\n    Checks if a cluster admin or database user exists.\n\n    If a database is specified: it will check for database user existence.\n    If a database is not specified: it will check for cluster admin existence.\n\n    name\n        User name\n\n    database\n        The database to check for the user to exist\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.user_exists <name>\n        salt '*' influxdb08.user_exists <name> <database>\n        salt '*' influxdb08.user_exists <name> <database> <user> <password> <host> <port>\n    "
    users = user_list(database, user, password, host, port)
    if not isinstance(users, list):
        return False
    for user in users:
        username = user.get('user', user.get('name'))
        if username:
            if username == name:
                return True
        else:
            log.warning('Could not find username in user: %s', user)
    return False

def user_create(name, passwd, database=None, user=None, password=None, host=None, port=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a cluster admin or a database user.\n\n    If a database is specified: it will create database user.\n    If a database is not specified: it will create a cluster admin.\n\n    name\n        User name for the new user to create\n\n    passwd\n        Password for the new user to create\n\n    database\n        The database to create the user in\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.user_create <name> <passwd>\n        salt '*' influxdb08.user_create <name> <passwd> <database>\n        salt '*' influxdb08.user_create <name> <passwd> <database> <user> <password> <host> <port>\n    "
    if user_exists(name, database, user, password, host, port):
        if database:
            log.info("User '%s' already exists for DB '%s'", name, database)
        else:
            log.info("Cluster admin '%s' already exists", name)
        return False
    client = _client(user=user, password=password, host=host, port=port)
    if not database:
        return client.add_cluster_admin(name, passwd)
    client.switch_database(database)
    return client.add_database_user(name, passwd)

def user_chpass(name, passwd, database=None, user=None, password=None, host=None, port=None):
    if False:
        while True:
            i = 10
    "\n    Change password for a cluster admin or a database user.\n\n    If a database is specified: it will update database user password.\n    If a database is not specified: it will update cluster admin password.\n\n    name\n        User name for whom to change the password\n\n    passwd\n        New password\n\n    database\n        The database on which to operate\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.user_chpass <name> <passwd>\n        salt '*' influxdb08.user_chpass <name> <passwd> <database>\n        salt '*' influxdb08.user_chpass <name> <passwd> <database> <user> <password> <host> <port>\n    "
    if not user_exists(name, database, user, password, host, port):
        if database:
            log.info("User '%s' does not exist for DB '%s'", name, database)
        else:
            log.info("Cluster admin '%s' does not exist", name)
        return False
    client = _client(user=user, password=password, host=host, port=port)
    if not database:
        return client.update_cluster_admin_password(name, passwd)
    client.switch_database(database)
    return client.update_database_user_password(name, passwd)

def user_remove(name, database=None, user=None, password=None, host=None, port=None):
    if False:
        while True:
            i = 10
    "\n    Remove a cluster admin or a database user.\n\n    If a database is specified: it will remove the database user.\n    If a database is not specified: it will remove the cluster admin.\n\n    name\n        User name to remove\n\n    database\n        The database to remove the user from\n\n    user\n        User name for the new user to delete\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.user_remove <name>\n        salt '*' influxdb08.user_remove <name> <database>\n        salt '*' influxdb08.user_remove <name> <database> <user> <password> <host> <port>\n    "
    if not user_exists(name, database, user, password, host, port):
        if database:
            log.info("User '%s' does not exist for DB '%s'", name, database)
        else:
            log.info("Cluster admin '%s' does not exist", name)
        return False
    client = _client(user=user, password=password, host=host, port=port)
    if not database:
        return client.delete_cluster_admin(name)
    client.switch_database(database)
    return client.delete_database_user(name)

def retention_policy_get(database, name, user=None, password=None, host=None, port=None):
    if False:
        while True:
            i = 10
    "\n    Get an existing retention policy.\n\n    database\n        The database to operate on.\n\n    name\n        Name of the policy to modify.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.retention_policy_get metrics default\n    "
    client = _client(user=user, password=password, host=host, port=port)
    for policy in client.get_list_retention_policies(database):
        if policy['name'] == name:
            return policy
    return None

def retention_policy_exists(database, name, user=None, password=None, host=None, port=None):
    if False:
        while True:
            i = 10
    "\n    Check if a retention policy exists.\n\n    database\n        The database to operate on.\n\n    name\n        Name of the policy to modify.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.retention_policy_exists metrics default\n    "
    policy = retention_policy_get(database, name, user, password, host, port)
    return policy is not None

def retention_policy_add(database, name, duration, replication, default=False, user=None, password=None, host=None, port=None):
    if False:
        i = 10
        return i + 15
    "\n    Add a retention policy.\n\n    database\n        The database to operate on.\n\n    name\n        Name of the policy to modify.\n\n    duration\n        How long InfluxDB keeps the data.\n\n    replication\n        How many copies of the data are stored in the cluster.\n\n    default\n        Whether this policy should be the default or not. Default is False.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb.retention_policy_add metrics default 1d 1\n    "
    client = _client(user=user, password=password, host=host, port=port)
    client.create_retention_policy(name, duration, replication, database, default)
    return True

def retention_policy_alter(database, name, duration, replication, default=False, user=None, password=None, host=None, port=None):
    if False:
        while True:
            i = 10
    "\n    Modify an existing retention policy.\n\n    database\n        The database to operate on.\n\n    name\n        Name of the policy to modify.\n\n    duration\n        How long InfluxDB keeps the data.\n\n    replication\n        How many copies of the data are stored in the cluster.\n\n    default\n        Whether this policy should be the default or not. Default is False.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.retention_policy_modify metrics default 1d 1\n    "
    client = _client(user=user, password=password, host=host, port=port)
    client.alter_retention_policy(name, database, duration, replication, default)
    return True

def query(database, query, time_precision='s', chunked=False, user=None, password=None, host=None, port=None):
    if False:
        print('Hello World!')
    "\n    Querying data\n\n    database\n        The database to query\n\n    query\n        Query to be executed\n\n    time_precision\n        Time precision to use ('s', 'm', or 'u')\n\n    chunked\n        Whether is chunked or not\n\n    user\n        The user to connect as\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.query <database> <query>\n        salt '*' influxdb08.query <database> <query> <time_precision> <chunked> <user> <password> <host> <port>\n    "
    client = _client(user=user, password=password, host=host, port=port)
    client.switch_database(database)
    return client.query(query, time_precision=time_precision, chunked=chunked)

def login_test(name, password, database=None, host=None, port=None):
    if False:
        i = 10
        return i + 15
    "\n    Checks if a credential pair can log in at all.\n\n    If a database is specified: it will check for database user existence.\n    If a database is not specified: it will check for cluster admin existence.\n\n    name\n        The user to connect as\n\n    password\n        The password of the user\n\n    database\n        The database to try to log in to\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' influxdb08.login_test <name>\n        salt '*' influxdb08.login_test <name> <database>\n        salt '*' influxdb08.login_test <name> <database> <user> <password> <host> <port>\n    "
    try:
        client = _client(user=name, password=password, host=host, port=port)
        client.get_list_database()
        return True
    except influxdb.influxdb08.client.InfluxDBClientError as e:
        if e.code == 401:
            return False
        else:
            raise