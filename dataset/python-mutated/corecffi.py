from __future__ import absolute_import, print_function
import sys
__all__ = ['get_version', 'get_header_version', 'supported_backends', 'recommended_backends', 'embeddable_backends', 'time', 'loop']
from zope.interface import implementer
from gevent._interfaces import ILoop
from gevent.libev import _corecffi
ffi = _corecffi.ffi
libev = _corecffi.lib
if hasattr(libev, 'vfd_open'):
    assert sys.platform.startswith('win'), 'vfd functions only needed on windows'
    vfd_open = libev.vfd_open
    vfd_free = libev.vfd_free
    vfd_get = libev.vfd_get
else:
    vfd_open = vfd_free = vfd_get = lambda fd: fd
libev.gevent_set_ev_alloc()
from gevent._ffi.loop import AbstractCallbacks
from gevent._ffi.loop import assign_standard_callbacks

class _Callbacks(AbstractCallbacks):

    def python_check_callback(self, *args):
        if False:
            print('Hello World!')
        (_loop, watcher_ptr, _events) = args
        AbstractCallbacks.python_check_callback(self, watcher_ptr)

    def _find_watcher_ptr_in_traceback(self, tb):
        if False:
            return 10
        if tb is not None:
            l = tb.tb_frame.f_locals
            if 'watcher_ptr' in l:
                return l['watcher_ptr']
            if 'args' in l and len(l['args']) == 3:
                return l['args'][1]
        return AbstractCallbacks._find_watcher_ptr_in_traceback(self, tb)

    def python_prepare_callback(self, _loop_ptr, watcher_ptr, _events):
        if False:
            while True:
                i = 10
        AbstractCallbacks.python_prepare_callback(self, watcher_ptr)

    def _find_loop_from_c_watcher(self, watcher_ptr):
        if False:
            return 10
        loop_handle = ffi.cast('struct ev_watcher*', watcher_ptr).data
        return self.from_handle(loop_handle)
_callbacks = assign_standard_callbacks(ffi, libev, _Callbacks)
UNDEF = libev.EV_UNDEF
NONE = libev.EV_NONE
READ = libev.EV_READ
WRITE = libev.EV_WRITE
TIMER = libev.EV_TIMER
PERIODIC = libev.EV_PERIODIC
SIGNAL = libev.EV_SIGNAL
CHILD = libev.EV_CHILD
STAT = libev.EV_STAT
IDLE = libev.EV_IDLE
PREPARE = libev.EV_PREPARE
CHECK = libev.EV_CHECK
EMBED = libev.EV_EMBED
FORK = libev.EV_FORK
CLEANUP = libev.EV_CLEANUP
ASYNC = libev.EV_ASYNC
CUSTOM = libev.EV_CUSTOM
ERROR = libev.EV_ERROR
READWRITE = libev.EV_READ | libev.EV_WRITE
MINPRI = libev.EV_MINPRI
MAXPRI = libev.EV_MAXPRI
BACKEND_PORT = libev.EVBACKEND_PORT
BACKEND_KQUEUE = libev.EVBACKEND_KQUEUE
BACKEND_EPOLL = libev.EVBACKEND_EPOLL
BACKEND_POLL = libev.EVBACKEND_POLL
BACKEND_SELECT = libev.EVBACKEND_SELECT
FORKCHECK = libev.EVFLAG_FORKCHECK
NOINOTIFY = libev.EVFLAG_NOINOTIFY
SIGNALFD = libev.EVFLAG_SIGNALFD
NOSIGMASK = libev.EVFLAG_NOSIGMASK
from gevent._ffi.loop import EVENTS
GEVENT_CORE_EVENTS = EVENTS

def get_version():
    if False:
        while True:
            i = 10
    return 'libev-%d.%02d' % (libev.ev_version_major(), libev.ev_version_minor())

def get_header_version():
    if False:
        while True:
            i = 10
    return 'libev-%d.%02d' % (libev.EV_VERSION_MAJOR, libev.EV_VERSION_MINOR)
_flags = [(libev.EVBACKEND_PORT, 'port'), (libev.EVBACKEND_KQUEUE, 'kqueue'), (libev.EVBACKEND_IOURING, 'linux_iouring'), (libev.EVBACKEND_LINUXAIO, 'linux_aio'), (libev.EVBACKEND_EPOLL, 'epoll'), (libev.EVBACKEND_POLL, 'poll'), (libev.EVBACKEND_SELECT, 'select'), (libev.EVFLAG_NOENV, 'noenv'), (libev.EVFLAG_FORKCHECK, 'forkcheck'), (libev.EVFLAG_SIGNALFD, 'signalfd'), (libev.EVFLAG_NOSIGMASK, 'nosigmask')]
_flags_str2int = dict(((string, flag) for (flag, string) in _flags))

def _flags_to_list(flags):
    if False:
        i = 10
        return i + 15
    result = []
    for (code, value) in _flags:
        if flags & code:
            result.append(value)
        flags &= ~code
        if not flags:
            break
    if flags:
        result.append(flags)
    return result
if sys.version_info[0] >= 3:
    basestring = (bytes, str)
    integer_types = (int,)
else:
    import __builtin__
    basestring = (__builtin__.basestring,)
    integer_types = (int, __builtin__.long)

def _flags_to_int(flags):
    if False:
        i = 10
        return i + 15
    if not flags:
        return 0
    if isinstance(flags, integer_types):
        return flags
    result = 0
    try:
        if isinstance(flags, basestring):
            flags = flags.split(',')
        for value in flags:
            value = value.strip().lower()
            if value:
                result |= _flags_str2int[value]
    except KeyError as ex:
        raise ValueError('Invalid backend or flag: %s\nPossible values: %s' % (ex, ', '.join(sorted(_flags_str2int.keys()))))
    return result

def _str_hex(flag):
    if False:
        while True:
            i = 10
    if isinstance(flag, integer_types):
        return hex(flag)
    return str(flag)

def _check_flags(flags):
    if False:
        return 10
    as_list = []
    flags &= libev.EVBACKEND_MASK
    if not flags:
        return
    if not flags & libev.EVBACKEND_ALL:
        raise ValueError('Invalid value for backend: 0x%x' % flags)
    if not flags & libev.ev_supported_backends():
        as_list = [_str_hex(x) for x in _flags_to_list(flags)]
        raise ValueError('Unsupported backend: %s' % '|'.join(as_list))

def supported_backends():
    if False:
        print('Hello World!')
    return _flags_to_list(libev.ev_supported_backends())

def recommended_backends():
    if False:
        print('Hello World!')
    return _flags_to_list(libev.ev_recommended_backends())

def embeddable_backends():
    if False:
        i = 10
        return i + 15
    return _flags_to_list(libev.ev_embeddable_backends())

def time():
    if False:
        print('Hello World!')
    return libev.ev_time()
from gevent._ffi.loop import AbstractLoop
from gevent.libev import watcher as _watchers
_events_to_str = _watchers._events_to_str

@implementer(ILoop)
class loop(AbstractLoop):
    approx_timer_resolution = 1e-05
    error_handler = None
    _CHECK_POINTER = 'struct ev_check *'
    _PREPARE_POINTER = 'struct ev_prepare *'
    _TIMER_POINTER = 'struct ev_timer *'

    def __init__(self, flags=None, default=None):
        if False:
            for i in range(10):
                print('nop')
        AbstractLoop.__init__(self, ffi, libev, _watchers, flags, default)
        self._default = bool(libev.ev_is_default_loop(self._ptr))

    def _init_loop(self, flags, default):
        if False:
            return 10
        c_flags = _flags_to_int(flags)
        _check_flags(c_flags)
        c_flags |= libev.EVFLAG_NOENV
        c_flags |= libev.EVFLAG_FORKCHECK
        if default is None:
            default = True
        if default:
            ptr = libev.gevent_ev_default_loop(c_flags)
            if not ptr:
                raise SystemError('ev_default_loop(%s) failed' % (c_flags,))
        else:
            ptr = libev.ev_loop_new(c_flags)
            if not ptr:
                raise SystemError('ev_loop_new(%s) failed' % (c_flags,))
        if default or SYSERR_CALLBACK is None:
            set_syserr_cb(self._handle_syserr)
        libev.ev_set_userdata(ptr, ptr)
        return ptr

    def _init_and_start_check(self):
        if False:
            print('Hello World!')
        libev.ev_check_init(self._check, libev.python_check_callback)
        self._check.data = self._handle_to_self
        libev.ev_check_start(self._ptr, self._check)
        self.unref()

    def _init_and_start_prepare(self):
        if False:
            print('Hello World!')
        libev.ev_prepare_init(self._prepare, libev.python_prepare_callback)
        libev.ev_prepare_start(self._ptr, self._prepare)
        self.unref()

    def _init_callback_timer(self):
        if False:
            print('Hello World!')
        libev.ev_timer_init(self._timer0, libev.gevent_noop, 0.0, 0.0)

    def _stop_callback_timer(self):
        if False:
            for i in range(10):
                print('nop')
        libev.ev_timer_stop(self._ptr, self._timer0)

    def _start_callback_timer(self):
        if False:
            while True:
                i = 10
        libev.ev_timer_start(self._ptr, self._timer0)

    def _stop_aux_watchers(self):
        if False:
            print('Hello World!')
        super(loop, self)._stop_aux_watchers()
        if libev.ev_is_active(self._prepare):
            self.ref()
            libev.ev_prepare_stop(self._ptr, self._prepare)
        if libev.ev_is_active(self._check):
            self.ref()
            libev.ev_check_stop(self._ptr, self._check)
        if libev.ev_is_active(self._timer0):
            libev.ev_timer_stop(self._timer0)

    def _setup_for_run_callback(self):
        if False:
            print('Hello World!')
        self.ref()

    def destroy(self):
        if False:
            while True:
                i = 10
        if self._ptr:
            super(loop, self).destroy()
            if globals()['SYSERR_CALLBACK'] == self._handle_syserr:
                set_syserr_cb(None)

    def _can_destroy_loop(self, ptr):
        if False:
            print('Hello World!')
        return libev.ev_userdata(ptr)

    def _destroy_loop(self, ptr):
        if False:
            i = 10
            return i + 15
        libev.ev_set_userdata(ptr, ffi.NULL)
        libev.ev_loop_destroy(ptr)
        libev.gevent_zero_prepare(self._prepare)
        libev.gevent_zero_check(self._check)
        libev.gevent_zero_timer(self._timer0)
        del self._prepare
        del self._check
        del self._timer0

    @property
    def MAXPRI(self):
        if False:
            i = 10
            return i + 15
        return libev.EV_MAXPRI

    @property
    def MINPRI(self):
        if False:
            while True:
                i = 10
        return libev.EV_MINPRI

    def _default_handle_error(self, context, type, value, tb):
        if False:
            for i in range(10):
                print('nop')
        super(loop, self)._default_handle_error(context, type, value, tb)
        libev.ev_break(self._ptr, libev.EVBREAK_ONE)

    def run(self, nowait=False, once=False):
        if False:
            for i in range(10):
                print('nop')
        flags = 0
        if nowait:
            flags |= libev.EVRUN_NOWAIT
        if once:
            flags |= libev.EVRUN_ONCE
        libev.ev_run(self._ptr, flags)

    def reinit(self):
        if False:
            print('Hello World!')
        libev.ev_loop_fork(self._ptr)

    def ref(self):
        if False:
            return 10
        libev.ev_ref(self._ptr)

    def unref(self):
        if False:
            return 10
        libev.ev_unref(self._ptr)

    def break_(self, how=libev.EVBREAK_ONE):
        if False:
            for i in range(10):
                print('nop')
        libev.ev_break(self._ptr, how)

    def verify(self):
        if False:
            i = 10
            return i + 15
        libev.ev_verify(self._ptr)

    def now(self):
        if False:
            for i in range(10):
                print('nop')
        return libev.ev_now(self._ptr)

    def update_now(self):
        if False:
            while True:
                i = 10
        libev.ev_now_update(self._ptr)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s at 0x%x %s>' % (self.__class__.__name__, id(self), self._format())

    @property
    def iteration(self):
        if False:
            for i in range(10):
                print('nop')
        return libev.ev_iteration(self._ptr)

    @property
    def depth(self):
        if False:
            while True:
                i = 10
        return libev.ev_depth(self._ptr)

    @property
    def backend_int(self):
        if False:
            for i in range(10):
                print('nop')
        return libev.ev_backend(self._ptr)

    @property
    def backend(self):
        if False:
            while True:
                i = 10
        backend = libev.ev_backend(self._ptr)
        for (key, value) in _flags:
            if key == backend:
                return value
        return backend

    @property
    def pendingcnt(self):
        if False:
            print('Hello World!')
        return libev.ev_pending_count(self._ptr)

    def closing_fd(self, fd):
        if False:
            i = 10
            return i + 15
        pending_before = libev.ev_pending_count(self._ptr)
        libev.ev_feed_fd_event(self._ptr, fd, 65535)
        pending_after = libev.ev_pending_count(self._ptr)
        return pending_after > pending_before
    if sys.platform != 'win32':

        def install_sigchld(self):
            if False:
                while True:
                    i = 10
            libev.gevent_install_sigchld_handler()

        def reset_sigchld(self):
            if False:
                i = 10
                return i + 15
            libev.gevent_reset_sigchld_handler()

    def fileno(self):
        if False:
            print('Hello World!')
        if self._ptr and LIBEV_EMBED:
            fd = self._ptr.backend_fd
            if fd >= 0:
                return fd

    @property
    def activecnt(self):
        if False:
            return 10
        if not self._ptr:
            raise ValueError('operation on destroyed loop')
        if LIBEV_EMBED:
            return self._ptr.activecnt
        return -1

@ffi.def_extern()
def _syserr_cb(msg):
    if False:
        print('Hello World!')
    try:
        msg = ffi.string(msg)
        SYSERR_CALLBACK(msg, ffi.errno)
    except:
        set_syserr_cb(None)
        raise

def set_syserr_cb(callback):
    if False:
        print('Hello World!')
    global SYSERR_CALLBACK
    if callback is None:
        libev.ev_set_syserr_cb(ffi.NULL)
        SYSERR_CALLBACK = None
    elif callable(callback):
        libev.ev_set_syserr_cb(libev._syserr_cb)
        SYSERR_CALLBACK = callback
    else:
        raise TypeError('Expected callable or None, got %r' % (callback,))
SYSERR_CALLBACK = None
LIBEV_EMBED = libev.LIBEV_EMBED