"""Selectors module.

This module allows high-level and efficient I/O multiplexing, built upon the
`select` module primitives.

The following code adapted from trollius.selectors.
"""
from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from collections import namedtuple
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from errno import EINTR
import math
import select
import sys
from kafka.vendor import six

def _wrap_error(exc, mapping, key):
    if False:
        i = 10
        return i + 15
    if key not in mapping:
        return
    new_err_cls = mapping[key]
    new_err = new_err_cls(*exc.args)
    if hasattr(exc, '__traceback__'):
        traceback = exc.__traceback__
    else:
        traceback = sys.exc_info()[2]
    six.reraise(new_err_cls, new_err, traceback)
EVENT_READ = 1 << 0
EVENT_WRITE = 1 << 1

def _fileobj_to_fd(fileobj):
    if False:
        i = 10
        return i + 15
    'Return a file descriptor from a file object.\n\n    Parameters:\n    fileobj -- file object or file descriptor\n\n    Returns:\n    corresponding file descriptor\n\n    Raises:\n    ValueError if the object is invalid\n    '
    if isinstance(fileobj, six.integer_types):
        fd = fileobj
    else:
        try:
            fd = int(fileobj.fileno())
        except (AttributeError, TypeError, ValueError):
            raise ValueError('Invalid file object: {0!r}'.format(fileobj))
    if fd < 0:
        raise ValueError('Invalid file descriptor: {0}'.format(fd))
    return fd
SelectorKey = namedtuple('SelectorKey', ['fileobj', 'fd', 'events', 'data'])
'Object used to associate a file object to its backing file descriptor,\nselected event mask and attached data.'

class _SelectorMapping(Mapping):
    """Mapping of file objects to selector keys."""

    def __init__(self, selector):
        if False:
            while True:
                i = 10
        self._selector = selector

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._selector._fd_to_key)

    def __getitem__(self, fileobj):
        if False:
            return 10
        try:
            fd = self._selector._fileobj_lookup(fileobj)
            return self._selector._fd_to_key[fd]
        except KeyError:
            raise KeyError('{0!r} is not registered'.format(fileobj))

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._selector._fd_to_key)

@six.add_metaclass(ABCMeta)
class BaseSelector(object):
    """Selector abstract base class.

    A selector supports registering file objects to be monitored for specific
    I/O events.

    A file object is a file descriptor or any object with a `fileno()` method.
    An arbitrary object can be attached to the file object, which can be used
    for example to store context information, a callback, etc.

    A selector can use various implementations (select(), poll(), epoll()...)
    depending on the platform. The default `Selector` class uses the most
    efficient implementation on the current platform.
    """

    @abstractmethod
    def register(self, fileobj, events, data=None):
        if False:
            for i in range(10):
                print('nop')
        'Register a file object.\n\n        Parameters:\n        fileobj -- file object or file descriptor\n        events  -- events to monitor (bitwise mask of EVENT_READ|EVENT_WRITE)\n        data    -- attached data\n\n        Returns:\n        SelectorKey instance\n\n        Raises:\n        ValueError if events is invalid\n        KeyError if fileobj is already registered\n        OSError if fileobj is closed or otherwise is unacceptable to\n                the underlying system call (if a system call is made)\n\n        Note:\n        OSError may or may not be raised\n        '
        raise NotImplementedError

    @abstractmethod
    def unregister(self, fileobj):
        if False:
            for i in range(10):
                print('nop')
        'Unregister a file object.\n\n        Parameters:\n        fileobj -- file object or file descriptor\n\n        Returns:\n        SelectorKey instance\n\n        Raises:\n        KeyError if fileobj is not registered\n\n        Note:\n        If fileobj is registered but has since been closed this does\n        *not* raise OSError (even if the wrapped syscall does)\n        '
        raise NotImplementedError

    def modify(self, fileobj, events, data=None):
        if False:
            for i in range(10):
                print('nop')
        'Change a registered file object monitored events or attached data.\n\n        Parameters:\n        fileobj -- file object or file descriptor\n        events  -- events to monitor (bitwise mask of EVENT_READ|EVENT_WRITE)\n        data    -- attached data\n\n        Returns:\n        SelectorKey instance\n\n        Raises:\n        Anything that unregister() or register() raises\n        '
        self.unregister(fileobj)
        return self.register(fileobj, events, data)

    @abstractmethod
    def select(self, timeout=None):
        if False:
            while True:
                i = 10
        "Perform the actual selection, until some monitored file objects are\n        ready or a timeout expires.\n\n        Parameters:\n        timeout -- if timeout > 0, this specifies the maximum wait time, in\n                   seconds\n                   if timeout <= 0, the select() call won't block, and will\n                   report the currently ready file objects\n                   if timeout is None, select() will block until a monitored\n                   file object becomes ready\n\n        Returns:\n        list of (key, events) for ready file objects\n        `events` is a bitwise mask of EVENT_READ|EVENT_WRITE\n        "
        raise NotImplementedError

    def close(self):
        if False:
            return 10
        'Close the selector.\n\n        This must be called to make sure that any underlying resource is freed.\n        '
        pass

    def get_key(self, fileobj):
        if False:
            while True:
                i = 10
        'Return the key associated to a registered file object.\n\n        Returns:\n        SelectorKey for this file object\n        '
        mapping = self.get_map()
        if mapping is None:
            raise RuntimeError('Selector is closed')
        try:
            return mapping[fileobj]
        except KeyError:
            raise KeyError('{0!r} is not registered'.format(fileobj))

    @abstractmethod
    def get_map(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a mapping of file objects to selector keys.'
        raise NotImplementedError

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        self.close()

class _BaseSelectorImpl(BaseSelector):
    """Base selector implementation."""

    def __init__(self):
        if False:
            print('Hello World!')
        self._fd_to_key = {}
        self._map = _SelectorMapping(self)

    def _fileobj_lookup(self, fileobj):
        if False:
            print('Hello World!')
        'Return a file descriptor from a file object.\n\n        This wraps _fileobj_to_fd() to do an exhaustive search in case\n        the object is invalid but we still have it in our map.  This\n        is used by unregister() so we can unregister an object that\n        was previously registered even if it is closed.  It is also\n        used by _SelectorMapping.\n        '
        try:
            return _fileobj_to_fd(fileobj)
        except ValueError:
            for key in self._fd_to_key.values():
                if key.fileobj is fileobj:
                    return key.fd
            raise

    def register(self, fileobj, events, data=None):
        if False:
            while True:
                i = 10
        if not events or events & ~(EVENT_READ | EVENT_WRITE):
            raise ValueError('Invalid events: {0!r}'.format(events))
        key = SelectorKey(fileobj, self._fileobj_lookup(fileobj), events, data)
        if key.fd in self._fd_to_key:
            raise KeyError('{0!r} (FD {1}) is already registered'.format(fileobj, key.fd))
        self._fd_to_key[key.fd] = key
        return key

    def unregister(self, fileobj):
        if False:
            i = 10
            return i + 15
        try:
            key = self._fd_to_key.pop(self._fileobj_lookup(fileobj))
        except KeyError:
            raise KeyError('{0!r} is not registered'.format(fileobj))
        return key

    def modify(self, fileobj, events, data=None):
        if False:
            while True:
                i = 10
        try:
            key = self._fd_to_key[self._fileobj_lookup(fileobj)]
        except KeyError:
            raise KeyError('{0!r} is not registered'.format(fileobj))
        if events != key.events:
            self.unregister(fileobj)
            key = self.register(fileobj, events, data)
        elif data != key.data:
            key = key._replace(data=data)
            self._fd_to_key[key.fd] = key
        return key

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self._fd_to_key.clear()
        self._map = None

    def get_map(self):
        if False:
            return 10
        return self._map

    def _key_from_fd(self, fd):
        if False:
            i = 10
            return i + 15
        'Return the key associated to a given file descriptor.\n\n        Parameters:\n        fd -- file descriptor\n\n        Returns:\n        corresponding key, or None if not found\n        '
        try:
            return self._fd_to_key[fd]
        except KeyError:
            return None

class SelectSelector(_BaseSelectorImpl):
    """Select-based selector."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(SelectSelector, self).__init__()
        self._readers = set()
        self._writers = set()

    def register(self, fileobj, events, data=None):
        if False:
            i = 10
            return i + 15
        key = super(SelectSelector, self).register(fileobj, events, data)
        if events & EVENT_READ:
            self._readers.add(key.fd)
        if events & EVENT_WRITE:
            self._writers.add(key.fd)
        return key

    def unregister(self, fileobj):
        if False:
            i = 10
            return i + 15
        key = super(SelectSelector, self).unregister(fileobj)
        self._readers.discard(key.fd)
        self._writers.discard(key.fd)
        return key
    if sys.platform == 'win32':

        def _select(self, r, w, _, timeout=None):
            if False:
                print('Hello World!')
            (r, w, x) = select.select(r, w, w, timeout)
            return (r, w + x, [])
    else:
        _select = staticmethod(select.select)

    def select(self, timeout=None):
        if False:
            while True:
                i = 10
        timeout = None if timeout is None else max(timeout, 0)
        ready = []
        try:
            (r, w, _) = self._select(self._readers, self._writers, [], timeout)
        except select.error as exc:
            if exc.args[0] == EINTR:
                return ready
            else:
                raise
        r = set(r)
        w = set(w)
        for fd in r | w:
            events = 0
            if fd in r:
                events |= EVENT_READ
            if fd in w:
                events |= EVENT_WRITE
            key = self._key_from_fd(fd)
            if key:
                ready.append((key, events & key.events))
        return ready
if hasattr(select, 'poll'):

    class PollSelector(_BaseSelectorImpl):
        """Poll-based selector."""

        def __init__(self):
            if False:
                return 10
            super(PollSelector, self).__init__()
            self._poll = select.poll()

        def register(self, fileobj, events, data=None):
            if False:
                return 10
            key = super(PollSelector, self).register(fileobj, events, data)
            poll_events = 0
            if events & EVENT_READ:
                poll_events |= select.POLLIN
            if events & EVENT_WRITE:
                poll_events |= select.POLLOUT
            self._poll.register(key.fd, poll_events)
            return key

        def unregister(self, fileobj):
            if False:
                print('Hello World!')
            key = super(PollSelector, self).unregister(fileobj)
            self._poll.unregister(key.fd)
            return key

        def select(self, timeout=None):
            if False:
                print('Hello World!')
            if timeout is None:
                timeout = None
            elif timeout <= 0:
                timeout = 0
            else:
                timeout = int(math.ceil(timeout * 1000.0))
            ready = []
            try:
                fd_event_list = self._poll.poll(timeout)
            except select.error as exc:
                if exc.args[0] == EINTR:
                    return ready
                else:
                    raise
            for (fd, event) in fd_event_list:
                events = 0
                if event & ~select.POLLIN:
                    events |= EVENT_WRITE
                if event & ~select.POLLOUT:
                    events |= EVENT_READ
                key = self._key_from_fd(fd)
                if key:
                    ready.append((key, events & key.events))
            return ready
if hasattr(select, 'epoll'):

    class EpollSelector(_BaseSelectorImpl):
        """Epoll-based selector."""

        def __init__(self):
            if False:
                return 10
            super(EpollSelector, self).__init__()
            self._epoll = select.epoll()

        def fileno(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._epoll.fileno()

        def register(self, fileobj, events, data=None):
            if False:
                print('Hello World!')
            key = super(EpollSelector, self).register(fileobj, events, data)
            epoll_events = 0
            if events & EVENT_READ:
                epoll_events |= select.EPOLLIN
            if events & EVENT_WRITE:
                epoll_events |= select.EPOLLOUT
            self._epoll.register(key.fd, epoll_events)
            return key

        def unregister(self, fileobj):
            if False:
                return 10
            key = super(EpollSelector, self).unregister(fileobj)
            try:
                self._epoll.unregister(key.fd)
            except IOError:
                pass
            return key

        def select(self, timeout=None):
            if False:
                return 10
            if timeout is None:
                timeout = -1
            elif timeout <= 0:
                timeout = 0
            else:
                timeout = math.ceil(timeout * 1000.0) * 0.001
            max_ev = max(len(self._fd_to_key), 1)
            ready = []
            try:
                fd_event_list = self._epoll.poll(timeout, max_ev)
            except IOError as exc:
                if exc.errno == EINTR:
                    return ready
                else:
                    raise
            for (fd, event) in fd_event_list:
                events = 0
                if event & ~select.EPOLLIN:
                    events |= EVENT_WRITE
                if event & ~select.EPOLLOUT:
                    events |= EVENT_READ
                key = self._key_from_fd(fd)
                if key:
                    ready.append((key, events & key.events))
            return ready

        def close(self):
            if False:
                print('Hello World!')
            self._epoll.close()
            super(EpollSelector, self).close()
if hasattr(select, 'devpoll'):

    class DevpollSelector(_BaseSelectorImpl):
        """Solaris /dev/poll selector."""

        def __init__(self):
            if False:
                return 10
            super(DevpollSelector, self).__init__()
            self._devpoll = select.devpoll()

        def fileno(self):
            if False:
                i = 10
                return i + 15
            return self._devpoll.fileno()

        def register(self, fileobj, events, data=None):
            if False:
                return 10
            key = super(DevpollSelector, self).register(fileobj, events, data)
            poll_events = 0
            if events & EVENT_READ:
                poll_events |= select.POLLIN
            if events & EVENT_WRITE:
                poll_events |= select.POLLOUT
            self._devpoll.register(key.fd, poll_events)
            return key

        def unregister(self, fileobj):
            if False:
                print('Hello World!')
            key = super(DevpollSelector, self).unregister(fileobj)
            self._devpoll.unregister(key.fd)
            return key

        def select(self, timeout=None):
            if False:
                i = 10
                return i + 15
            if timeout is None:
                timeout = None
            elif timeout <= 0:
                timeout = 0
            else:
                timeout = math.ceil(timeout * 1000.0)
            ready = []
            try:
                fd_event_list = self._devpoll.poll(timeout)
            except OSError as exc:
                if exc.errno == EINTR:
                    return ready
                else:
                    raise
            for (fd, event) in fd_event_list:
                events = 0
                if event & ~select.POLLIN:
                    events |= EVENT_WRITE
                if event & ~select.POLLOUT:
                    events |= EVENT_READ
                key = self._key_from_fd(fd)
                if key:
                    ready.append((key, events & key.events))
            return ready

        def close(self):
            if False:
                while True:
                    i = 10
            self._devpoll.close()
            super(DevpollSelector, self).close()
if hasattr(select, 'kqueue'):

    class KqueueSelector(_BaseSelectorImpl):
        """Kqueue-based selector."""

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super(KqueueSelector, self).__init__()
            self._kqueue = select.kqueue()

        def fileno(self):
            if False:
                return 10
            return self._kqueue.fileno()

        def register(self, fileobj, events, data=None):
            if False:
                return 10
            key = super(KqueueSelector, self).register(fileobj, events, data)
            if events & EVENT_READ:
                kev = select.kevent(key.fd, select.KQ_FILTER_READ, select.KQ_EV_ADD)
                self._kqueue.control([kev], 0, 0)
            if events & EVENT_WRITE:
                kev = select.kevent(key.fd, select.KQ_FILTER_WRITE, select.KQ_EV_ADD)
                self._kqueue.control([kev], 0, 0)
            return key

        def unregister(self, fileobj):
            if False:
                for i in range(10):
                    print('nop')
            key = super(KqueueSelector, self).unregister(fileobj)
            if key.events & EVENT_READ:
                kev = select.kevent(key.fd, select.KQ_FILTER_READ, select.KQ_EV_DELETE)
                try:
                    self._kqueue.control([kev], 0, 0)
                except OSError:
                    pass
            if key.events & EVENT_WRITE:
                kev = select.kevent(key.fd, select.KQ_FILTER_WRITE, select.KQ_EV_DELETE)
                try:
                    self._kqueue.control([kev], 0, 0)
                except OSError:
                    pass
            return key

        def select(self, timeout=None):
            if False:
                i = 10
                return i + 15
            timeout = None if timeout is None else max(timeout, 0)
            max_ev = len(self._fd_to_key)
            ready = []
            try:
                kev_list = self._kqueue.control(None, max_ev, timeout)
            except OSError as exc:
                if exc.errno == EINTR:
                    return ready
                else:
                    raise
            for kev in kev_list:
                fd = kev.ident
                flag = kev.filter
                events = 0
                if flag == select.KQ_FILTER_READ:
                    events |= EVENT_READ
                if flag == select.KQ_FILTER_WRITE:
                    events |= EVENT_WRITE
                key = self._key_from_fd(fd)
                if key:
                    ready.append((key, events & key.events))
            return ready

        def close(self):
            if False:
                i = 10
                return i + 15
            self._kqueue.close()
            super(KqueueSelector, self).close()
if 'KqueueSelector' in globals():
    DefaultSelector = KqueueSelector
elif 'EpollSelector' in globals():
    DefaultSelector = EpollSelector
elif 'DevpollSelector' in globals():
    DefaultSelector = DevpollSelector
elif 'PollSelector' in globals():
    DefaultSelector = PollSelector
else:
    DefaultSelector = SelectSelector