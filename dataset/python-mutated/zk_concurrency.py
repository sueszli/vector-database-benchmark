"""
Concurrency controls in zookeeper
=========================================================================

:depends: kazoo
:configuration: See :py:mod:`salt.modules.zookeeper` for setup instructions.

This module allows you to acquire and release a slot. This is primarily useful
for ensureing that no more than N hosts take a specific action at once. This can
also be used to coordinate between masters.
"""
import logging
import sys
try:
    from socket import gethostname
    import kazoo.client
    import kazoo.recipe.barrier
    import kazoo.recipe.lock
    import kazoo.recipe.party
    from kazoo.exceptions import CancelledError, NoNodeError
    from kazoo.retry import ForceRetryError

    class _Semaphore(kazoo.recipe.lock.Semaphore):

        def __init__(self, client, path, identifier=None, max_leases=1, ephemeral_lease=True):
            if False:
                while True:
                    i = 10
            identifier = identifier or gethostname()
            kazoo.recipe.lock.Semaphore.__init__(self, client, path, identifier=identifier, max_leases=max_leases)
            self.ephemeral_lease = ephemeral_lease
            if not self.ephemeral_lease:
                try:
                    for child in self.client.get_children(self.path):
                        try:
                            (data, stat) = self.client.get(self.path + '/' + child)
                            if identifier == data.decode('utf-8'):
                                self.create_path = self.path + '/' + child
                                self.is_acquired = True
                                break
                        except NoNodeError:
                            pass
                except NoNodeError:
                    pass

        def _get_lease(self, data=None):
            if False:
                print('Hello World!')
            if self._session_expired:
                raise ForceRetryError('Retry on session loss at top')
            if self.cancelled:
                raise CancelledError('Semaphore cancelled')
            children = self.client.get_children(self.path, self._watch_lease_change)
            if len(children) < self.max_leases:
                self.client.create(self.create_path, self.data, ephemeral=self.ephemeral_lease)
            if self.client.exists(self.create_path):
                self.is_acquired = True
            else:
                self.is_acquired = False
            return self.is_acquired
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
__virtualname__ = 'zk_concurrency'

def __virtual__():
    if False:
        i = 10
        return i + 15
    if not HAS_DEPS:
        return (False, 'Module zk_concurrency: dependencies failed')
    __context__['semaphore_map'] = {}
    return __virtualname__

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
            default_acl = [__salt__['zookeeper.make_digest_acl'](**acl) for acl in default_acl]
        else:
            default_acl = [__salt__['zookeeper.make_digest_acl'](**default_acl)]
    __context__.setdefault('zkconnection', {}).setdefault(profile or hosts, kazoo.client.KazooClient(hosts=hosts, default_acl=default_acl, auth_data=auth_data))
    if not __context__['zkconnection'][profile or hosts].connected:
        __context__['zkconnection'][profile or hosts].start()
    return __context__['zkconnection'][profile or hosts]

def lock_holders(path, zk_hosts=None, identifier=None, max_concurrency=1, timeout=None, ephemeral_lease=False, profile=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        print('Hello World!')
    '\n    Return an un-ordered list of lock holders\n\n    path\n        The path in zookeeper where the lock is\n\n    zk_hosts\n        zookeeper connect string\n\n    identifier\n        Name to identify this minion, if unspecified defaults to hostname\n\n    max_concurrency\n        Maximum number of lock holders\n\n    timeout\n        timeout to wait for the lock. A None timeout will block forever\n\n    ephemeral_lease\n        Whether the locks in zookeper should be ephemeral\n\n    Example:\n\n    .. code-block:: bash\n\n        salt minion zk_concurrency.lock_holders /lock/path host1:1234,host2:1234\n    '
    zk = _get_zk_conn(profile=profile, hosts=zk_hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    if path not in __context__['semaphore_map']:
        __context__['semaphore_map'][path] = _Semaphore(zk, path, identifier, max_leases=max_concurrency, ephemeral_lease=ephemeral_lease)
    return __context__['semaphore_map'][path].lease_holders()

def lock(path, zk_hosts=None, identifier=None, max_concurrency=1, timeout=None, ephemeral_lease=False, force=False, profile=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        i = 10
        return i + 15
    '\n    Get lock (with optional timeout)\n\n    path\n        The path in zookeeper where the lock is\n\n    zk_hosts\n        zookeeper connect string\n\n    identifier\n        Name to identify this minion, if unspecified defaults to the hostname\n\n    max_concurrency\n        Maximum number of lock holders\n\n    timeout\n        timeout to wait for the lock. A None timeout will block forever\n\n    ephemeral_lease\n        Whether the locks in zookeper should be ephemeral\n\n    force\n        Forcibly acquire the lock regardless of available slots\n\n    Example:\n\n    .. code-block:: bash\n\n        salt minion zk_concurrency.lock /lock/path host1:1234,host2:1234\n    '
    zk = _get_zk_conn(profile=profile, hosts=zk_hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    if path not in __context__['semaphore_map']:
        __context__['semaphore_map'][path] = _Semaphore(zk, path, identifier, max_leases=max_concurrency, ephemeral_lease=ephemeral_lease)
    if force:
        __context__['semaphore_map'][path].assured_path = True
        __context__['semaphore_map'][path].max_leases = sys.maxint
    if timeout:
        logging.info('Acquiring lock %s with timeout=%s', path, timeout)
        __context__['semaphore_map'][path].acquire(timeout=timeout)
    else:
        logging.info('Acquiring lock %s with no timeout', path)
        __context__['semaphore_map'][path].acquire()
    return __context__['semaphore_map'][path].is_acquired

def unlock(path, zk_hosts=None, identifier=None, max_concurrency=1, ephemeral_lease=False, scheme=None, profile=None, username=None, password=None, default_acl=None):
    if False:
        i = 10
        return i + 15
    '\n    Remove lease from semaphore\n\n    path\n        The path in zookeeper where the lock is\n\n    zk_hosts\n        zookeeper connect string\n\n    identifier\n        Name to identify this minion, if unspecified defaults to hostname\n\n    max_concurrency\n        Maximum number of lock holders\n\n    timeout\n        timeout to wait for the lock. A None timeout will block forever\n\n    ephemeral_lease\n        Whether the locks in zookeper should be ephemeral\n\n    Example:\n\n    .. code-block:: bash\n\n        salt minion zk_concurrency.unlock /lock/path host1:1234,host2:1234\n    '
    zk = _get_zk_conn(profile=profile, hosts=zk_hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    if path not in __context__['semaphore_map']:
        __context__['semaphore_map'][path] = _Semaphore(zk, path, identifier, max_leases=max_concurrency, ephemeral_lease=ephemeral_lease)
    if path in __context__['semaphore_map']:
        __context__['semaphore_map'][path].release()
        del __context__['semaphore_map'][path]
        return True
    else:
        logging.error('Unable to find lease for path %s', path)
        return False

def party_members(path, zk_hosts=None, min_nodes=1, blocking=False, profile=None, scheme=None, username=None, password=None, default_acl=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the List of identifiers in a particular party, optionally waiting for the\n    specified minimum number of nodes (min_nodes) to appear\n\n    path\n        The path in zookeeper where the lock is\n\n    zk_hosts\n        zookeeper connect string\n\n    min_nodes\n        The minimum number of nodes expected to be present in the party\n\n    blocking\n        The boolean indicating if we need to block until min_nodes are available\n\n    Example:\n\n    .. code-block:: bash\n\n        salt minion zk_concurrency.party_members /lock/path host1:1234,host2:1234\n        salt minion zk_concurrency.party_members /lock/path host1:1234,host2:1234 min_nodes=3 blocking=True\n    '
    zk = _get_zk_conn(profile=profile, hosts=zk_hosts, scheme=scheme, username=username, password=password, default_acl=default_acl)
    party = kazoo.recipe.party.ShallowParty(zk, path)
    if blocking:
        barrier = kazoo.recipe.barrier.DoubleBarrier(zk, path, min_nodes)
        barrier.enter()
        party = kazoo.recipe.party.ShallowParty(zk, path)
        barrier.leave()
    return list(party)