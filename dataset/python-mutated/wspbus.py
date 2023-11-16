"""An implementation of the Web Site Process Bus.

This module is completely standalone, depending only on the stdlib.

Web Site Process Bus
--------------------

A Bus object is used to contain and manage site-wide behavior:
daemonization, HTTP server start/stop, process reload, signal handling,
drop privileges, PID file management, logging for all of these,
and many more.

In addition, a Bus object provides a place for each web framework
to register code that runs in response to site-wide events (like
process start and stop), or which controls or otherwise interacts with
the site-wide components mentioned above. For example, a framework which
uses file-based templates would add known template filenames to an
autoreload component.

Ideally, a Bus object will be flexible enough to be useful in a variety
of invocation scenarios:

 1. The deployer starts a site from the command line via a
    framework-neutral deployment script; applications from multiple frameworks
    are mixed in a single site. Command-line arguments and configuration
    files are used to define site-wide components such as the HTTP server,
    WSGI component graph, autoreload behavior, signal handling, etc.
 2. The deployer starts a site via some other process, such as Apache;
    applications from multiple frameworks are mixed in a single site.
    Autoreload and signal handling (from Python at least) are disabled.
 3. The deployer starts a site via a framework-specific mechanism;
    for example, when running tests, exploring tutorials, or deploying
    single applications from a single framework. The framework controls
    which site-wide components are enabled as it sees fit.

The Bus object in this package uses topic-based publish-subscribe
messaging to accomplish all this. A few topic channels are built in
('start', 'stop', 'exit', 'graceful', 'log', and 'main'). Frameworks and
site containers are free to define their own. If a message is sent to a
channel that has not been defined or has no listeners, there is no effect.

In general, there should only ever be a single Bus object per process.
Frameworks and site containers share a single Bus object by publishing
messages and subscribing listeners.

The Bus object works as a finite state machine which models the current
state of the process. Bus methods move it from one state to another;
those methods then publish to subscribed listeners on the channel for
the new state.::

                        O
                        |
                        V
       STOPPING --> STOPPED --> EXITING -> X
          A   A         |
          |    \\___     |
          |        \\    |
          |         V   V
        STARTED <-- STARTING

"""
import atexit
try:
    import ctypes
except ImportError:
    'Google AppEngine is shipped without ctypes\n\n    :seealso: http://stackoverflow.com/a/6523777/70170\n    '
    ctypes = None
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
_startup_cwd = os.getcwd()

class ChannelFailures(Exception):
    """Exception raised during errors on Bus.publish()."""
    delimiter = '\n'

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Initialize ChannelFailures errors wrapper.'
        super(ChannelFailures, self).__init__(*args, **kwargs)
        self._exceptions = list()

    def handle_exception(self):
        if False:
            while True:
                i = 10
        'Append the current exception to self.'
        self._exceptions.append(sys.exc_info()[1])

    def get_instances(self):
        if False:
            i = 10
            return i + 15
        'Return a list of seen exception instances.'
        return self._exceptions[:]

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Render the list of errors, which happened in channel.'
        exception_strings = map(repr, self.get_instances())
        return self.delimiter.join(exception_strings)
    __repr__ = __str__

    def __bool__(self):
        if False:
            while True:
                i = 10
        'Determine whether any error happened in channel.'
        return bool(self._exceptions)
    __nonzero__ = __bool__

class _StateEnum(object):

    class State(object):
        name = None

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'states.%s' % self.name

    def __setattr__(self, key, value):
        if False:
            return 10
        if isinstance(value, self.State):
            value.name = key
        object.__setattr__(self, key, value)
states = _StateEnum()
states.STOPPED = states.State()
states.STARTING = states.State()
states.STARTED = states.State()
states.STOPPING = states.State()
states.EXITING = states.State()
try:
    import fcntl
except ImportError:
    max_files = 0
else:
    try:
        max_files = os.sysconf('SC_OPEN_MAX')
    except AttributeError:
        max_files = 1024

class Bus(object):
    """Process state-machine and messenger for HTTP site deployment.

    All listeners for a given channel are guaranteed to be called even
    if others at the same channel fail. Each failure is logged, but
    execution proceeds on to the next listener. The only way to stop all
    processing from inside a listener is to raise SystemExit and stop the
    whole server.
    """
    states = states
    state = states.STOPPED
    execv = False
    max_cloexec_files = max_files

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initialize pub/sub bus.'
        self.execv = False
        self.state = states.STOPPED
        channels = ('start', 'stop', 'exit', 'graceful', 'log', 'main')
        self.listeners = dict(((channel, set()) for channel in channels))
        self._priorities = {}

    def subscribe(self, channel, callback=None, priority=None):
        if False:
            print('Hello World!')
        'Add the given callback at the given channel (if not present).\n\n        If callback is None, return a partial suitable for decorating\n        the callback.\n        '
        if callback is None:
            return functools.partial(self.subscribe, channel, priority=priority)
        ch_listeners = self.listeners.setdefault(channel, set())
        ch_listeners.add(callback)
        if priority is None:
            priority = getattr(callback, 'priority', 50)
        self._priorities[channel, callback] = priority

    def unsubscribe(self, channel, callback):
        if False:
            while True:
                i = 10
        'Discard the given callback (if present).'
        listeners = self.listeners.get(channel)
        if listeners and callback in listeners:
            listeners.discard(callback)
            del self._priorities[channel, callback]

    def publish(self, channel, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return output of all subscribers for the given channel.'
        if channel not in self.listeners:
            return []
        exc = ChannelFailures()
        output = []
        raw_items = ((self._priorities[channel, listener], listener) for listener in self.listeners[channel])
        items = sorted(raw_items, key=operator.itemgetter(0))
        for (priority, listener) in items:
            try:
                output.append(listener(*args, **kwargs))
            except KeyboardInterrupt:
                raise
            except SystemExit:
                e = sys.exc_info()[1]
                if exc and e.code == 0:
                    e.code = 1
                raise
            except Exception:
                exc.handle_exception()
                if channel == 'log':
                    pass
                else:
                    self.log('Error in %r listener %r' % (channel, listener), level=40, traceback=True)
        if exc:
            raise exc
        return output

    def _clean_exit(self):
        if False:
            while True:
                i = 10
        'Assert that the Bus is not running in atexit handler callback.'
        if self.state != states.EXITING:
            warnings.warn('The main thread is exiting, but the Bus is in the %r state; shutting it down automatically now. You must either call bus.block() after start(), or call bus.exit() before the main thread exits.' % self.state, RuntimeWarning)
            self.exit()

    def start(self):
        if False:
            while True:
                i = 10
        'Start all services.'
        atexit.register(self._clean_exit)
        self.state = states.STARTING
        self.log('Bus STARTING')
        try:
            self.publish('start')
            self.state = states.STARTED
            self.log('Bus STARTED')
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.log('Shutting down due to error in start listener:', level=40, traceback=True)
            e_info = sys.exc_info()[1]
            try:
                self.exit()
            except Exception:
                pass
            raise e_info

    def exit(self):
        if False:
            while True:
                i = 10
        'Stop all services and prepare to exit the process.'
        exitstate = self.state
        EX_SOFTWARE = 70
        try:
            self.stop()
            self.state = states.EXITING
            self.log('Bus EXITING')
            self.publish('exit')
            self.log('Bus EXITED')
        except Exception:
            os._exit(EX_SOFTWARE)
        if exitstate == states.STARTING:
            os._exit(EX_SOFTWARE)

    def restart(self):
        if False:
            return 10
        'Restart the process (may close connections).\n\n        This method does not restart the process from the calling thread;\n        instead, it stops the bus and asks the main thread to call execv.\n        '
        self.execv = True
        self.exit()

    def graceful(self):
        if False:
            print('Hello World!')
        'Advise all services to reload.'
        self.log('Bus graceful')
        self.publish('graceful')

    def block(self, interval=0.1):
        if False:
            i = 10
            return i + 15
        'Wait for the EXITING state, KeyboardInterrupt or SystemExit.\n\n        This function is intended to be called only by the main thread.\n        After waiting for the EXITING state, it also waits for all threads\n        to terminate, and then calls os.execv if self.execv is True. This\n        design allows another thread to call bus.restart, yet have the main\n        thread perform the actual execv call (required on some platforms).\n        '
        try:
            self.wait(states.EXITING, interval=interval, channel='main')
        except (KeyboardInterrupt, IOError):
            self.log('Keyboard Interrupt: shutting down bus')
            self.exit()
        except SystemExit:
            self.log('SystemExit raised: shutting down bus')
            self.exit()
            raise
        self.log('Waiting for child threads to terminate...')
        for t in threading.enumerate():
            if t != threading.current_thread() and (not isinstance(t, threading._MainThread)) and (not t.daemon):
                self.log('Waiting for thread %s.' % t.name)
                t.join()
        if self.execv:
            self._do_execv()

    def wait(self, state, interval=0.1, channel=None):
        if False:
            return 10
        'Poll for the given state(s) at intervals; publish to channel.'
        states = set(always_iterable(state))
        while self.state not in states:
            time.sleep(interval)
            self.publish(channel)

    def _do_execv(self):
        if False:
            while True:
                i = 10
        "Re-execute the current process.\n\n        This must be called from the main thread, because certain platforms\n        (OS X) don't allow execv to be called in a child thread very well.\n        "
        try:
            args = self._get_true_argv()
        except NotImplementedError:
            "It's probably win32 or GAE"
            args = [sys.executable] + self._get_interpreter_argv() + sys.argv
        self.log('Re-spawning %s' % ' '.join(args))
        self._extend_pythonpath(os.environ)
        if sys.platform[:4] == 'java':
            from _systemrestart import SystemRestart
            raise SystemRestart
        else:
            if sys.platform == 'win32':
                args = ['"%s"' % arg for arg in args]
            os.chdir(_startup_cwd)
            if self.max_cloexec_files:
                self._set_cloexec()
            os.execv(sys.executable, args)

    @staticmethod
    def _get_interpreter_argv():
        if False:
            while True:
                i = 10
        "Retrieve current Python interpreter's arguments.\n\n        Returns empty tuple in case of frozen mode, uses built-in arguments\n        reproduction function otherwise.\n\n        Frozen mode is possible for the app has been packaged into a binary\n        executable using py2exe. In this case the interpreter's arguments are\n        already built-in into that executable.\n\n        :seealso: https://github.com/cherrypy/cherrypy/issues/1526\n        Ref: https://pythonhosted.org/PyInstaller/runtime-information.html\n        "
        return [] if getattr(sys, 'frozen', False) else subprocess._args_from_interpreter_flags()

    @staticmethod
    def _get_true_argv():
        if False:
            for i in range(10):
                print('nop')
        'Retrieve all real arguments of the python interpreter.\n\n        ...even those not listed in ``sys.argv``\n\n        :seealso: http://stackoverflow.com/a/28338254/595220\n        :seealso: http://stackoverflow.com/a/6683222/595220\n        :seealso: http://stackoverflow.com/a/28414807/595220\n        '
        try:
            char_p = ctypes.c_wchar_p
            argv = ctypes.POINTER(char_p)()
            argc = ctypes.c_int()
            ctypes.pythonapi.Py_GetArgcArgv(ctypes.byref(argc), ctypes.byref(argv))
            _argv = argv[:argc.value]
            (argv_len, is_command, is_module) = (len(_argv), False, False)
            try:
                m_ind = _argv.index('-m')
                if m_ind < argv_len - 1 and _argv[m_ind + 1] in ('-c', '-m'):
                    "\n                    In some older Python versions `-m`'s argument may be\n                    substituted with `-c`, not `-m`\n                    "
                    is_module = True
            except (IndexError, ValueError):
                m_ind = None
            try:
                c_ind = _argv.index('-c')
                if c_ind < argv_len - 1 and _argv[c_ind + 1] == '-c':
                    is_command = True
            except (IndexError, ValueError):
                c_ind = None
            if is_module:
                "It's containing `-m -m` sequence of arguments"
                if is_command and c_ind < m_ind:
                    "There's `-c -c` before `-m`"
                    raise RuntimeError("Cannot reconstruct command from '-c'. Ref: https://github.com/cherrypy/cherrypy/issues/1545")
                original_module = sys.argv[0]
                if not os.access(original_module, os.R_OK):
                    "There's no such module exist"
                    raise AttributeError("{} doesn't seem to be a module accessible by current user".format(original_module))
                del _argv[m_ind:m_ind + 2]
                _argv.insert(m_ind, original_module)
            elif is_command:
                "It's containing just `-c -c` sequence of arguments"
                raise RuntimeError("Cannot reconstruct command from '-c'. Ref: https://github.com/cherrypy/cherrypy/issues/1545")
        except AttributeError:
            "It looks Py_GetArgcArgv is completely absent in some environments\n\n            It is known, that there's no Py_GetArgcArgv in MS Windows and\n            ``ctypes`` module is completely absent in Google AppEngine\n\n            :seealso: https://github.com/cherrypy/cherrypy/issues/1506\n            :seealso: https://github.com/cherrypy/cherrypy/issues/1512\n            :ref: http://bit.ly/2gK6bXK\n            "
            raise NotImplementedError
        else:
            return _argv

    @staticmethod
    def _extend_pythonpath(env):
        if False:
            while True:
                i = 10
        'Prepend current working dir to PATH environment variable if needed.\n\n        If sys.path[0] is an empty string, the interpreter was likely\n        invoked with -m and the effective path is about to change on\n        re-exec.  Add the current directory to $PYTHONPATH to ensure\n        that the new process sees the same path.\n\n        This issue cannot be addressed in the general case because\n        Python cannot reliably reconstruct the\n        original command line (http://bugs.python.org/issue14208).\n\n        (This idea filched from tornado.autoreload)\n        '
        path_prefix = '.' + os.pathsep
        existing_path = env.get('PYTHONPATH', '')
        needs_patch = sys.path[0] == '' and (not existing_path.startswith(path_prefix))
        if needs_patch:
            env['PYTHONPATH'] = path_prefix + existing_path

    def _set_cloexec(self):
        if False:
            for i in range(10):
                print('nop')
        'Set the CLOEXEC flag on all open files (except stdin/out/err).\n\n        If self.max_cloexec_files is an integer (the default), then on\n        platforms which support it, it represents the max open files setting\n        for the operating system. This function will be called just before\n        the process is restarted via os.execv() to prevent open files\n        from persisting into the new process.\n\n        Set self.max_cloexec_files to 0 to disable this behavior.\n        '
        for fd in range(3, self.max_cloexec_files):
            try:
                flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            except IOError:
                continue
            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

    def stop(self):
        if False:
            i = 10
            return i + 15
        'Stop all services.'
        self.state = states.STOPPING
        self.log('Bus STOPPING')
        self.publish('stop')
        self.state = states.STOPPED
        self.log('Bus STOPPED')

    def start_with_callback(self, func, args=None, kwargs=None):
        if False:
            i = 10
            return i + 15
        "Start 'func' in a new thread T, then start self (and return T)."
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        args = (func,) + args

        def _callback(func, *a, **kw):
            if False:
                print('Hello World!')
            self.wait(states.STARTED)
            func(*a, **kw)
        t = threading.Thread(target=_callback, args=args, kwargs=kwargs)
        t.name = 'Bus Callback ' + t.name
        t.start()
        self.start()
        return t

    def log(self, msg='', level=20, traceback=False):
        if False:
            i = 10
            return i + 15
        'Log the given message. Append the last traceback if requested.'
        if traceback:
            msg += '\n' + ''.join(_traceback.format_exception(*sys.exc_info()))
        self.publish('log', msg, level)
bus = Bus()