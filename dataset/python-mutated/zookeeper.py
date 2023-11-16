"""
Zookeeper Module
~~~~~~~~~~~~~~~~
:maintainer:    SaltStack
:maturity:      new
:platform:      all
:depends:       kazoo

.. versionadded:: 2018.3.0

Configuration
=============

:configuration: This module is not usable until the following are specified
    either in a pillar or in the minion's config file:

    .. code-block:: yaml

        zookeeper:
          hosts: zoo1,zoo2,zoo3
          default_acl:
            - username: daniel
              password: test
              read: true
              write: true
              create: true
              delete: true
              admin: true
          username: daniel
          password: test

    If configuration for multiple zookeeper environments is required, they can
    be set up as different configuration profiles. For example:

    .. code-block:: yaml

        zookeeper:
          prod:
            hosts: zoo1,zoo2,zoo3
            default_acl:
              - username: daniel
                password: test
                read: true
                write: true
                create: true
                delete: true
                admin: true
            username: daniel
            password: test
          dev:
            hosts:
              - dev1
              - dev2
              - dev3
            default_acl:
              - username: daniel
                password: test
                read: true
                write: true
                create: true
                delete: true
                admin: true
            username: daniel
            password: test
"""
import salt.utils.stringutils
try:
    import kazoo.client
    import kazoo.security
    HAS_KAZOO = True
except ImportError:
    HAS_KAZOO = False
__virtualname__ = 'zookeeper'

def __virtual__():
    if False:
        i = 10
        return i + 15
    if HAS_KAZOO:
        return __virtualname__
    return (False, 'Missing dependency: kazoo')

def _get_zk_conn(profile=None, **connection_args):
    if False:
        i = 10
        return i + 15
    if profile:
        prefix = 'zookeeper:' + profile
    else:
        prefix = 'zookeeper'

    def get(key, default=None):
        if False:
            i = 10
            return i + 15
        '\n        look in connection_args first, then default to config file\n        '
        return connection_args.get(key) or __salt__['config.get'](':'.join([prefix, key]), default)
    hosts = get('hosts', '127.0.0.1:2181')
    scheme = get('scheme', None)
    username = get('username', None)
    password = get('password', None)
    default_acl = get('default_acl', None)
    if isinstance(hosts, list):
        hosts = ','.join(hosts)
    if username is not None and password is not None and (scheme is None):
        scheme = 'digest'
    auth_data = None
    if scheme and username and password:
        auth_data = [(scheme, ':'.join([username, password]))]
    if default_acl is not None:
        if isinstance(default_acl, list):
            default_acl = [make_digest_acl(**acl) for acl in default_acl]
        else:
            default_acl = [make_digest_acl(**default_acl)]
    __context__.setdefault('zkconnection', {}).setdefault(profile or hosts, kazoo.client.KazooClient(hosts=hosts, default_acl=default_acl, auth_data=auth_data))
    if not __context__['zkconnection'][profile or hosts].connected:
        __context__['zkconnection'][profile or hosts].start()
    return __context__['zkconnection'][profile or hosts]

def create(path, value='', acls=None, ephemeral=False, sequence=False, makepath=False, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        i = 10
        return i + 15
    "\n    Create Znode\n\n    path\n        path of znode to create\n\n    value\n        value to assign to znode (Default: '')\n\n    acls\n        list of acl dictionaries to be assigned (Default: None)\n\n    ephemeral\n        indicate node is ephemeral (Default: False)\n\n    sequence\n        indicate node is suffixed with a unique index (Default: False)\n\n    makepath\n        Create parent paths if they do not exist (Default: False)\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: '127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: 'digest')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.create /test/name daniel profile=prod\n\n    "
    if acls is None:
        acls = []
    acls = [make_digest_acl(**acl) for acl in acls]
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    return conn.create(path, salt.utils.stringutils.to_bytes(value), acls, ephemeral, sequence, makepath)

def ensure_path(path, acls=None, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        return 10
    "\n    Ensure Znode path exists\n\n    path\n        Parent path to create\n\n    acls\n        list of acls dictionaries to be assigned (Default: None)\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: '127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: 'digest')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.ensure_path /test/name profile=prod\n\n    "
    if acls is None:
        acls = []
    acls = [make_digest_acl(**acl) for acl in acls]
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    return conn.ensure_path(path, acls)

def exists(path, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        return 10
    "\n    Check if path exists\n\n    path\n        path to check\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: '127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: 'digest')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.exists /test/name profile=prod\n\n    "
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    return bool(conn.exists(path))

def get(path, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get value saved in znode\n\n    path\n        path to check\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: '127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: 'digest')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.get /test/name profile=prod\n\n    "
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    (ret, _) = conn.get(path)
    return salt.utils.stringutils.to_str(ret)

def get_children(path, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        while True:
            i = 10
    "\n    Get children in znode path\n\n    path\n        path to check\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: '127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: 'digest')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.get_children /test profile=prod\n\n    "
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    ret = conn.get_children(path)
    return ret or []

def set(path, value, version=-1, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        print('Hello World!')
    "\n    Update znode with new value\n\n    path\n        znode to update\n\n    value\n        value to set in znode\n\n    version\n        only update znode if version matches (Default: -1 (always matches))\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: '127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: 'digest')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.set /test/name gtmanfred profile=prod\n\n    "
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    return conn.set(path, salt.utils.stringutils.to_bytes(value), version=version)

def get_acls(path, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        return 10
    "\n    Get acls on a znode\n\n    path\n        path to znode\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: '127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: 'digest')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.get_acls /test/name profile=prod\n\n    "
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    return conn.get_acls(path)[0]

def set_acls(path, acls, version=-1, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        return 10
    '\n    Set acls on a znode\n\n    path\n        path to znode\n\n    acls\n        list of acl dictionaries to set on the znode\n\n    version\n        only set acls if version matches (Default: -1 (always matches))\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: \'127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: \'digest\')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.set_acls /test/name acls=\'[{"username": "gtmanfred", "password": "test", "all": True}]\' profile=prod\n\n    '
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    if acls is None:
        acls = []
    acls = [make_digest_acl(**acl) for acl in acls]
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    return conn.set_acls(path, acls, version)

def delete(path, version=-1, recursive=False, profile=None, hosts=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        i = 10
        return i + 15
    "\n    Delete znode\n\n    path\n        path to znode\n\n    version\n        only delete if version matches (Default: -1 (always matches))\n\n    profile\n        Configured Zookeeper profile to authenticate with (Default: None)\n\n    hosts\n        Lists of Zookeeper Hosts (Default: '127.0.0.1:2181)\n\n    scheme\n        Scheme to authenticate with (Default: 'digest')\n\n    username\n        Username to authenticate (Default: None)\n\n    password\n        Password to authenticate (Default: None)\n\n    default_acl\n        Default acls to assign if a node is created in this connection (Default: None)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.delete /test/name profile=prod\n\n    "
    conn = _get_zk_conn(profile=profile, hosts=hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    return conn.delete(path, version, recursive)

def make_digest_acl(username, password, read=False, write=False, create=False, delete=False, admin=False, allperms=False):
    if False:
        while True:
            i = 10
    '\n    Generate acl object\n\n    .. note:: This is heavily used in the zookeeper state and probably is not useful as a cli module\n\n    username\n        username of acl\n\n    password\n        plain text password of acl\n\n    read\n        read acl\n\n    write\n        write acl\n\n    create\n        create acl\n\n    delete\n        delete acl\n\n    admin\n        admin acl\n\n    allperms\n        set all other acls to True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt minion1 zookeeper.make_digest_acl username=daniel password=mypass allperms=True\n    '
    return kazoo.security.make_digest_acl(username, password, read, write, create, delete, admin, allperms)