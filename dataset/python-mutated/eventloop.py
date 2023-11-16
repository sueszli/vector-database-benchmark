from __future__ import absolute_import, division, print_function, with_statement
import os
import time
import socket
import select
import errno
import logging
from collections import defaultdict
from ssshare.shadowsocks import shell
__all__ = ['EventLoop', 'POLL_NULL', 'POLL_IN', 'POLL_OUT', 'POLL_ERR', 'POLL_HUP', 'POLL_NVAL', 'EVENT_NAMES']
POLL_NULL = 0
POLL_IN = 1
POLL_OUT = 4
POLL_ERR = 8
POLL_HUP = 16
POLL_NVAL = 32
EVENT_NAMES = {POLL_NULL: 'POLL_NULL', POLL_IN: 'POLL_IN', POLL_OUT: 'POLL_OUT', POLL_ERR: 'POLL_ERR', POLL_HUP: 'POLL_HUP', POLL_NVAL: 'POLL_NVAL'}
TIMEOUT_PRECISION = 2

class KqueueLoop(object):
    MAX_EVENTS = 1024

    def __init__(self):
        if False:
            while True:
                i = 10
        self._kqueue = select.kqueue()
        self._fds = {}

    def _control(self, fd, mode, flags):
        if False:
            for i in range(10):
                print('nop')
        events = []
        if mode & POLL_IN:
            events.append(select.kevent(fd, select.KQ_FILTER_READ, flags))
        if mode & POLL_OUT:
            events.append(select.kevent(fd, select.KQ_FILTER_WRITE, flags))
        for e in events:
            self._kqueue.control([e], 0)

    def poll(self, timeout):
        if False:
            return 10
        if timeout < 0:
            timeout = None
        events = self._kqueue.control(None, KqueueLoop.MAX_EVENTS, timeout)
        results = defaultdict(lambda : POLL_NULL)
        for e in events:
            fd = e.ident
            if e.filter == select.KQ_FILTER_READ:
                results[fd] |= POLL_IN
            elif e.filter == select.KQ_FILTER_WRITE:
                results[fd] |= POLL_OUT
        return results.items()

    def register(self, fd, mode):
        if False:
            print('Hello World!')
        self._fds[fd] = mode
        self._control(fd, mode, select.KQ_EV_ADD)

    def unregister(self, fd):
        if False:
            i = 10
            return i + 15
        self._control(fd, self._fds[fd], select.KQ_EV_DELETE)
        del self._fds[fd]

    def modify(self, fd, mode):
        if False:
            return 10
        self.unregister(fd)
        self.register(fd, mode)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self._kqueue.close()

class SelectLoop(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._r_list = set()
        self._w_list = set()
        self._x_list = set()

    def poll(self, timeout):
        if False:
            while True:
                i = 10
        (r, w, x) = select.select(self._r_list, self._w_list, self._x_list, timeout)
        results = defaultdict(lambda : POLL_NULL)
        for p in [(r, POLL_IN), (w, POLL_OUT), (x, POLL_ERR)]:
            for fd in p[0]:
                results[fd] |= p[1]
        return results.items()

    def register(self, fd, mode):
        if False:
            return 10
        if mode & POLL_IN:
            self._r_list.add(fd)
        if mode & POLL_OUT:
            self._w_list.add(fd)
        if mode & POLL_ERR:
            self._x_list.add(fd)

    def unregister(self, fd):
        if False:
            for i in range(10):
                print('nop')
        if fd in self._r_list:
            self._r_list.remove(fd)
        if fd in self._w_list:
            self._w_list.remove(fd)
        if fd in self._x_list:
            self._x_list.remove(fd)

    def modify(self, fd, mode):
        if False:
            for i in range(10):
                print('nop')
        self.unregister(fd)
        self.register(fd, mode)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class EventLoop(object):

    def __init__(self):
        if False:
            return 10
        if hasattr(select, 'epoll'):
            self._impl = select.epoll()
            model = 'epoll'
        elif hasattr(select, 'kqueue'):
            self._impl = KqueueLoop()
            model = 'kqueue'
        elif hasattr(select, 'select'):
            self._impl = SelectLoop()
            model = 'select'
        else:
            raise Exception('can not find any available functions in select package')
        self._fdmap = {}
        self._last_time = time.time()
        self._periodic_callbacks = []
        self._stopping = False
        logging.debug('using event model: %s', model)

    def poll(self, timeout=None):
        if False:
            while True:
                i = 10
        events = self._impl.poll(timeout)
        return [(self._fdmap[fd][0], fd, event) for (fd, event) in events]

    def add(self, f, mode, handler):
        if False:
            return 10
        fd = f.fileno()
        self._fdmap[fd] = (f, handler)
        self._impl.register(fd, mode)

    def remove(self, f):
        if False:
            while True:
                i = 10
        fd = f.fileno()
        del self._fdmap[fd]
        self._impl.unregister(fd)

    def removefd(self, fd):
        if False:
            i = 10
            return i + 15
        del self._fdmap[fd]
        self._impl.unregister(fd)

    def add_periodic(self, callback):
        if False:
            i = 10
            return i + 15
        self._periodic_callbacks.append(callback)

    def remove_periodic(self, callback):
        if False:
            while True:
                i = 10
        self._periodic_callbacks.remove(callback)

    def modify(self, f, mode):
        if False:
            while True:
                i = 10
        fd = f.fileno()
        self._impl.modify(fd, mode)

    def stop(self):
        if False:
            print('Hello World!')
        self._stopping = True

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        events = []
        while not self._stopping:
            asap = False
            try:
                events = self.poll(TIMEOUT_PRECISION)
            except (OSError, IOError) as e:
                if errno_from_exception(e) in (errno.EPIPE, errno.EINTR):
                    asap = True
                    logging.debug('poll:%s', e)
                else:
                    logging.error('poll:%s', e)
                    import traceback
                    traceback.print_exc()
                    continue
            handle = False
            for (sock, fd, event) in events:
                handler = self._fdmap.get(fd, None)
                if handler is not None:
                    handler = handler[1]
                    try:
                        handle = handler.handle_event(sock, fd, event) or handle
                    except (OSError, IOError) as e:
                        shell.print_exception(e)
            now = time.time()
            if asap or now - self._last_time >= TIMEOUT_PRECISION:
                for callback in self._periodic_callbacks:
                    callback()
                self._last_time = now
            if events and (not handle):
                time.sleep(0.001)

    def __del__(self):
        if False:
            return 10
        self._impl.close()

def errno_from_exception(e):
    if False:
        i = 10
        return i + 15
    'Provides the errno from an Exception object.\n\n    There are cases that the errno attribute was not set so we pull\n    the errno out of the args but if someone instantiates an Exception\n    without any args you will get a tuple error. So this function\n    abstracts all that behavior to give you a safe way to get the\n    errno.\n    '
    if hasattr(e, 'errno'):
        return e.errno
    elif e.args:
        return e.args[0]
    else:
        return None

def get_sock_error(sock):
    if False:
        while True:
            i = 10
    error_number = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    return socket.error(error_number, os.strerror(error_number))