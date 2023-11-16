from __future__ import absolute_import
import six
from oslo_config import cfg
from tooz import coordination
from tooz import locking
from tooz.coordination import GroupNotCreated
from tooz.coordination import MemberNotJoined
from st2common import log as logging
from st2common.util import system_info
LOG = logging.getLogger(__name__)
COORDINATOR = None
__all__ = ['configured', 'get_coordinator', 'get_coordinator_if_set', 'get_member_id', 'coordinator_setup', 'coordinator_teardown']

class NoOpLock(locking.Lock):

    def __init__(self, name='noop'):
        if False:
            for i in range(10):
                print('nop')
        super(NoOpLock, self).__init__(name=name)

    def acquire(self, blocking=True):
        if False:
            return 10
        return True

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def heartbeat(self):
        if False:
            print('Hello World!')
        return True

class NoOpAsyncResult(object):
    """
    In most scenarios, tooz library returns an async result, a future and this
    class wrapper is here to correctly mimic tooz API and behavior.
    """

    def __init__(self, result=None):
        if False:
            print('Hello World!')
        self._result = result

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        return self._result

class NoOpDriver(coordination.CoordinationDriver):
    """
    Tooz driver where each operation is a no-op.

    This driver is used if coordination service is not configured.
    """
    groups = {}

    def __init__(self, member_id, parsed_url=None, options=None):
        if False:
            while True:
                i = 10
        super(NoOpDriver, self).__init__(member_id, parsed_url, options)

    @classmethod
    def stop(cls):
        if False:
            while True:
                i = 10
        cls.groups = {}

    def watch_join_group(self, group_id, callback):
        if False:
            while True:
                i = 10
        self._hooks_join_group[group_id].append(callback)

    def unwatch_join_group(self, group_id, callback):
        if False:
            while True:
                i = 10
        return None

    def watch_leave_group(self, group_id, callback):
        if False:
            for i in range(10):
                print('nop')
        return None

    def unwatch_leave_group(self, group_id, callback):
        if False:
            return 10
        return None

    def watch_elected_as_leader(self, group_id, callback):
        if False:
            print('Hello World!')
        return None

    def unwatch_elected_as_leader(self, group_id, callback):
        if False:
            return 10
        return None

    @staticmethod
    def stand_down_group_leader(group_id):
        if False:
            print('Hello World!')
        return None

    @classmethod
    def create_group(cls, group_id):
        if False:
            for i in range(10):
                print('nop')
        cls.groups[group_id] = {'members': {}}
        return NoOpAsyncResult()

    @classmethod
    def get_groups(cls):
        if False:
            while True:
                i = 10
        return NoOpAsyncResult(result=cls.groups.keys())

    @classmethod
    def join_group(cls, group_id, capabilities=''):
        if False:
            while True:
                i = 10
        member_id = get_member_id()
        cls.groups[group_id]['members'][member_id] = {'capabilities': capabilities}
        return NoOpAsyncResult()

    @classmethod
    def leave_group(cls, group_id):
        if False:
            return 10
        member_id = get_member_id()
        try:
            members = cls.groups[group_id]['members']
        except KeyError:
            raise GroupNotCreated(group_id)
        try:
            del members[member_id]
        except KeyError:
            raise MemberNotJoined(group_id, member_id)
        return NoOpAsyncResult()

    @classmethod
    def delete_group(cls, group_id):
        if False:
            i = 10
            return i + 15
        del cls.groups[group_id]
        return NoOpAsyncResult()

    @classmethod
    def get_members(cls, group_id):
        if False:
            while True:
                i = 10
        try:
            member_ids = cls.groups[group_id]['members'].keys()
        except KeyError:
            raise GroupNotCreated('Group doesnt exist')
        return NoOpAsyncResult(result=member_ids)

    @classmethod
    def get_member_capabilities(cls, group_id, member_id):
        if False:
            print('Hello World!')
        member_capabiliteis = cls.groups[group_id]['members'][member_id]['capabilities']
        return NoOpAsyncResult(result=member_capabiliteis)

    @staticmethod
    def update_capabilities(group_id, capabilities):
        if False:
            while True:
                i = 10
        return None

    @staticmethod
    def get_leader(group_id):
        if False:
            return 10
        return None

    @staticmethod
    def get_lock(name):
        if False:
            return 10
        return NoOpLock(name='noop')

def configured():
    if False:
        return 10
    '\n    Return True if the coordination service is properly configured.\n\n    :rtype: ``bool``\n    '
    backend_configured = cfg.CONF.coordination.url is not None
    mock_backend = backend_configured and (cfg.CONF.coordination.url.startswith('zake') or cfg.CONF.coordination.url.startswith('file'))
    return backend_configured and (not mock_backend)

def get_driver_name() -> str:
    if False:
        while True:
            i = 10
    '\n    Return coordination driver name (aka protocol part from the URI / URL).\n    '
    url = cfg.CONF.coordination.url
    if not url:
        return None
    driver_name = url.split('://')[0]
    return driver_name

def coordinator_setup(start_heart=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets up the client for the coordination service.\n\n    URL examples for connection:\n        zake://\n        file:///tmp\n        redis://username:password@host:port\n        mysql://username:password@host:port/dbname\n    '
    url = cfg.CONF.coordination.url
    lock_timeout = cfg.CONF.coordination.lock_timeout
    member_id = get_member_id()
    if url:
        coordinator = coordination.get_coordinator(url, member_id, lock_timeout=lock_timeout)
    else:
        coordinator = NoOpDriver(member_id)
    coordinator.start(start_heart=start_heart)
    return coordinator

def coordinator_teardown(coordinator=None):
    if False:
        while True:
            i = 10
    if coordinator:
        coordinator.stop()

def get_coordinator(start_heart=True, use_cache=True):
    if False:
        return 10
    '\n    :param start_heart: True to start heartbeating process.\n    :type start_heart: ``bool``\n\n    :param use_cache: True to use cached coordinator instance. False should only be used in tests.\n    :type use_cache: ``bool``\n    '
    global COORDINATOR
    if not configured():
        LOG.warn('Coordination backend is not configured. Code paths which use coordination service will use best effort approach and race conditions are possible.')
    if not use_cache:
        return coordinator_setup(start_heart=start_heart)
    if not COORDINATOR:
        COORDINATOR = coordinator_setup(start_heart=start_heart)
        LOG.debug('Initializing and caching new coordinator instance: %s' % str(COORDINATOR))
    else:
        LOG.debug('Using cached coordinator instance: %s' % str(COORDINATOR))
    return COORDINATOR

def get_coordinator_if_set():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a coordinator instance if one has been initialized, None otherwise.\n    '
    global COORDINATOR
    return COORDINATOR

def get_member_id():
    if False:
        i = 10
        return i + 15
    '\n    Retrieve member if for the current process.\n\n    :rtype: ``bytes``\n    '
    proc_info = system_info.get_process_info()
    member_id = six.b('%s_%d' % (proc_info['hostname'], proc_info['pid']))
    return member_id

def get_group_id(service):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(service, six.binary_type):
        group_id = service.encode('utf-8')
    else:
        group_id = service
    return group_id