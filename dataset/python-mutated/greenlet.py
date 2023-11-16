from __future__ import absolute_import, print_function, division
from sys import _getframe as sys_getframe
from sys import exc_info as sys_exc_info
from weakref import ref as wref
from greenlet import greenlet
from greenlet import GreenletExit
from gevent._compat import reraise
from gevent._compat import PYPY as _PYPY
from gevent._tblib import dump_traceback
from gevent._tblib import load_traceback
from gevent.exceptions import InvalidSwitchError
from gevent._hub_primitives import iwait_on_objects as iwait
from gevent._hub_primitives import wait_on_objects as wait
from gevent.timeout import Timeout
from gevent._config import config as GEVENT_CONFIG
from gevent._util import readproperty
from gevent._hub_local import get_hub_noargs as get_hub
from gevent import _waiter
__all__ = ['Greenlet', 'joinall', 'killall']
locals()['getcurrent'] = __import__('greenlet').getcurrent
locals()['greenlet_init'] = lambda : None
locals()['Waiter'] = _waiter.Waiter
locals()['get_my_hub'] = lambda s: s.parent
locals()['get_generic_parent'] = lambda s: s.parent
locals()['Gevent_PyFrame_GetCode'] = lambda frame: frame.f_code
locals()['Gevent_PyFrame_GetLineNumber'] = lambda frame: frame.f_lineno
locals()['Gevent_PyFrame_GetBack'] = lambda frame: frame.f_back
if _PYPY:
    import _continuation
    _continulet = _continuation.continulet

class SpawnedLink(object):
    """
    A wrapper around link that calls it in another greenlet.

    Can be called only from main loop.
    """
    __slots__ = ['callback']

    def __init__(self, callback):
        if False:
            print('Hello World!')
        if not callable(callback):
            raise TypeError('Expected callable: %r' % (callback,))
        self.callback = callback

    def __call__(self, source):
        if False:
            return 10
        g = greenlet(self.callback, get_hub())
        g.switch(source)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.callback)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.callback == getattr(other, 'callback', other)

    def __str__(self):
        if False:
            print('Hello World!')
        return str(self.callback)

    def __repr__(self):
        if False:
            print('Hello World!')
        return repr(self.callback)

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        assert item != 'callback'
        return getattr(self.callback, item)

class SuccessSpawnedLink(SpawnedLink):
    """A wrapper around link that calls it in another greenlet only if source succeed.

    Can be called only from main loop.
    """
    __slots__ = []

    def __call__(self, source):
        if False:
            i = 10
            return i + 15
        if source.successful():
            return SpawnedLink.__call__(self, source)

class FailureSpawnedLink(SpawnedLink):
    """A wrapper around link that calls it in another greenlet only if source failed.

    Can be called only from main loop.
    """
    __slots__ = []

    def __call__(self, source):
        if False:
            print('Hello World!')
        if not source.successful():
            return SpawnedLink.__call__(self, source)

class _Frame(object):
    __slots__ = ('f_code', 'f_lineno', 'f_back')

    def __init__(self):
        if False:
            return 10
        self.f_code = None
        self.f_back = None
        self.f_lineno = 0

    @property
    def f_globals(self):
        if False:
            i = 10
            return i + 15
        return None

def _extract_stack(limit):
    if False:
        print('Hello World!')
    try:
        frame = sys_getframe()
    except ValueError:
        frame = None
    newest_Frame = None
    newer_Frame = None
    while limit and frame is not None:
        limit -= 1
        older_Frame = _Frame()
        older_Frame.f_code = Gevent_PyFrame_GetCode(frame)
        older_Frame.f_lineno = Gevent_PyFrame_GetLineNumber(frame)
        if newer_Frame is not None:
            newer_Frame.f_back = older_Frame
        newer_Frame = older_Frame
        if newest_Frame is None:
            newest_Frame = newer_Frame
        frame = Gevent_PyFrame_GetBack(frame)
    return newest_Frame
_greenlet__init__ = greenlet.__init__

class Greenlet(greenlet):
    """
    A light-weight cooperatively-scheduled execution unit.
    """
    spawning_stack_limit = 10

    def __init__(self, run=None, *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        :param args: The arguments passed to the ``run`` function.\n        :param kwargs: The keyword arguments passed to the ``run`` function.\n        :keyword callable run: The callable object to run. If not given, this object's\n            `_run` method will be invoked (typically defined by subclasses).\n\n        .. versionchanged:: 1.1b1\n            The ``run`` argument to the constructor is now verified to be a callable\n            object. Previously, passing a non-callable object would fail after the greenlet\n            was spawned.\n\n        .. versionchanged:: 1.3b1\n           The ``GEVENT_TRACK_GREENLET_TREE`` configuration value may be set to\n           a false value to disable ``spawn_tree_locals``, ``spawning_greenlet``,\n           and ``spawning_stack``. The first two will be None in that case, and the\n           latter will be empty.\n\n        .. versionchanged:: 1.5\n           Greenlet objects are now more careful to verify that their ``parent`` is really\n           a gevent hub, raising a ``TypeError`` earlier instead of an ``AttributeError`` later.\n\n        .. versionchanged:: 20.12.1\n           Greenlet objects now function as context managers. Exiting the ``with`` suite\n           ensures that the greenlet has completed by :meth:`joining <join>`\n           the greenlet (blocking, with\n           no timeout). If the body of the suite raises an exception, the greenlet is\n           :meth:`killed <kill>` with the default arguments and not joined in that case.\n        "
        _greenlet__init__(self, None, get_hub())
        if run is not None:
            self._run = run
        if not callable(self._run):
            raise TypeError('The run argument or self._run must be callable')
        self.args = args
        self.kwargs = kwargs
        self.value = None
        self._start_event = None
        self._notifier = None
        self._formatted_info = None
        self._links = []
        self._ident = None
        self._exc_info = None
        if GEVENT_CONFIG.track_greenlet_tree:
            spawner = getcurrent()
            self.spawning_greenlet = wref(spawner)
            try:
                self.spawn_tree_locals = spawner.spawn_tree_locals
            except AttributeError:
                self.spawn_tree_locals = {}
                if get_generic_parent(spawner) is not None:
                    spawner.spawn_tree_locals = self.spawn_tree_locals
            self.spawning_stack = _extract_stack(self.spawning_stack_limit)
        else:
            self.spawning_greenlet = None
            self.spawn_tree_locals = None
            self.spawning_stack = None

    def _get_minimal_ident(self):
        if False:
            return 10
        hub = get_my_hub(self)
        reg = hub.ident_registry
        return reg.get_ident(self)

    @property
    def minimal_ident(self):
        if False:
            i = 10
            return i + 15
        "\n        A small, unique non-negative integer that identifies this object.\n\n        This is similar to :attr:`threading.Thread.ident` (and `id`)\n        in that as long as this object is alive, no other greenlet *in\n        this hub* will have the same id, but it makes a stronger\n        guarantee that the assigned values will be small and\n        sequential. Sometime after this object has died, the value\n        will be available for reuse.\n\n        To get ids that are unique across all hubs, combine this with\n        the hub's (``self.parent``) ``minimal_ident``.\n\n        Accessing this property from threads other than the thread running\n        this greenlet is not defined.\n\n        .. versionadded:: 1.3a2\n\n        "
        if self._ident is None:
            self._ident = self._get_minimal_ident()
        return self._ident

    @readproperty
    def name(self):
        if False:
            i = 10
            return i + 15
        "\n        The greenlet name. By default, a unique name is constructed using\n        the :attr:`minimal_ident`. You can assign a string to this\n        value to change it. It is shown in the `repr` of this object if it\n        has been assigned to or if the `minimal_ident` has already been generated.\n\n        .. versionadded:: 1.3a2\n        .. versionchanged:: 1.4\n           Stop showing generated names in the `repr` when the ``minimal_ident``\n           hasn't been requested. This reduces overhead and may be less confusing,\n           since ``minimal_ident`` can get reused.\n        "
        return 'Greenlet-%d' % (self.minimal_ident,)

    def _raise_exception(self):
        if False:
            return 10
        reraise(*self.exc_info)

    @property
    def loop(self):
        if False:
            i = 10
            return i + 15
        hub = get_my_hub(self)
        return hub.loop

    def __nonzero__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._start_event is not None and self._exc_info is None
    try:
        __bool__ = __nonzero__
    except NameError:
        pass
    if _PYPY:

        @property
        def dead(self):
            if False:
                return 10
            'Boolean indicating that the greenlet is dead and will not run again.'
            if self._greenlet__main:
                return False
            if self.__start_cancelled_by_kill() or self.__started_but_aborted():
                return True
            return self._greenlet__started and (not _continulet.is_pending(self))
    else:

        @property
        def dead(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Boolean indicating that the greenlet is dead and will not run again.\n\n            This is true if:\n\n            1. We were never started, but were :meth:`killed <kill>`\n               immediately after creation (not possible with :meth:`spawn`); OR\n            2. We were started, but were killed before running; OR\n            3. We have run and terminated (by raising an exception out of the\n               started function or by reaching the end of the started function).\n            '
            return self.__start_cancelled_by_kill() or self.__started_but_aborted() or greenlet.dead.__get__(self)

    def __never_started_or_killed(self):
        if False:
            for i in range(10):
                print('nop')
        return self._start_event is None

    def __start_pending(self):
        if False:
            for i in range(10):
                print('nop')
        return self._start_event is not None and (self._start_event.pending or getattr(self._start_event, 'active', False))

    def __start_cancelled_by_kill(self):
        if False:
            for i in range(10):
                print('nop')
        return self._start_event is _cancelled_start_event

    def __start_completed(self):
        if False:
            for i in range(10):
                print('nop')
        return self._start_event is _start_completed_event

    def __started_but_aborted(self):
        if False:
            i = 10
            return i + 15
        return not self.__never_started_or_killed() and (not self.__start_cancelled_by_kill()) and (not self.__start_completed()) and (not self.__start_pending())

    def __cancel_start(self):
        if False:
            return 10
        if self._start_event is None:
            self._start_event = _cancelled_start_event
        self._start_event.stop()
        self._start_event.close()

    def __handle_death_before_start(self, args):
        if False:
            while True:
                i = 10
        if self._exc_info is None and self.dead:
            if len(args) == 1:
                arg = args[0]
                if isinstance(arg, type) and issubclass(arg, BaseException):
                    args = (arg, arg(), None)
                else:
                    args = (type(arg), arg, None)
            elif not args:
                args = (GreenletExit, GreenletExit(), None)
            if not issubclass(args[0], BaseException):
                print('RANDOM CRAP', args)
                import traceback
                traceback.print_stack()
                args = (BaseException, BaseException(args), None)
            assert issubclass(args[0], BaseException)
            self.__report_error(args)

    @property
    def started(self):
        if False:
            return 10
        return bool(self)

    def ready(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a true value if and only if the greenlet has finished\n        execution.\n\n        .. versionchanged:: 1.1\n            This function is only guaranteed to return true or false *values*, not\n            necessarily the literal constants ``True`` or ``False``.\n        '
        return self.dead or self._exc_info is not None

    def successful(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a true value if and only if the greenlet has finished execution\n        successfully, that is, without raising an error.\n\n        .. tip:: A greenlet that has been killed with the default\n            :class:`GreenletExit` exception is considered successful.\n            That is, ``GreenletExit`` is not considered an error.\n\n        .. note:: This function is only guaranteed to return true or false *values*,\n              not necessarily the literal constants ``True`` or ``False``.\n        '
        return self._exc_info is not None and self._exc_info[1] is None

    def __repr__(self):
        if False:
            while True:
                i = 10
        classname = self.__class__.__name__
        if 'name' not in self.__dict__ and self._ident is None:
            name = ' '
        else:
            name = ' "%s" ' % (self.name,)
        result = '<%s%sat %s' % (classname, name, hex(id(self)))
        formatted = self._formatinfo()
        if formatted:
            result += ': ' + formatted
        return result + '>'

    def _formatinfo(self):
        if False:
            print('Hello World!')
        info = self._formatted_info
        if info is not None:
            return info
        func = self._run
        im_self = getattr(func, '__self__', None)
        if im_self is self:
            funcname = '_run'
        elif im_self is not None:
            funcname = repr(func)
        else:
            funcname = getattr(func, '__name__', '') or repr(func)
        result = funcname
        args = []
        if self.args:
            args = [repr(x)[:50] for x in self.args]
        if self.kwargs:
            args.extend(['%s=%s' % (key, repr(value)[:50]) for (key, value) in self.kwargs.items()])
        if args:
            result += '(' + ', '.join(args) + ')'
        self._formatted_info = result
        return result

    @property
    def exception(self):
        if False:
            while True:
                i = 10
        '\n        Holds the exception instance raised by the function if the\n        greenlet has finished with an error. Otherwise ``None``.\n        '
        return self._exc_info[1] if self._exc_info is not None else None

    @property
    def exc_info(self):
        if False:
            return 10
        '\n        Holds the exc_info three-tuple raised by the function if the\n        greenlet finished with an error. Otherwise a false value.\n\n        .. note:: This is a provisional API and may change.\n\n        .. versionadded:: 1.1\n        '
        ei = self._exc_info
        if ei is not None and ei[0] is not None:
            return (ei[0], ei[1], load_traceback(ei[2]) if ei[2] else None)

    def throw(self, *args):
        if False:
            print('Hello World!')
        'Immediately switch into the greenlet and raise an exception in it.\n\n        Should only be called from the HUB, otherwise the current greenlet is left unscheduled forever.\n        To raise an exception in a safe manner from any greenlet, use :meth:`kill`.\n\n        If a greenlet was started but never switched to yet, then also\n        a) cancel the event that will start it\n        b) fire the notifications as if an exception was raised in a greenlet\n        '
        self.__cancel_start()
        try:
            if not self.dead:
                greenlet.throw(self, *args)
        finally:
            self.__handle_death_before_start(args)

    def start(self):
        if False:
            return 10
        'Schedule the greenlet to run in this loop iteration'
        if self._start_event is None:
            _call_spawn_callbacks(self)
            hub = get_my_hub(self)
            self._start_event = hub.loop.run_callback(self.switch)

    def start_later(self, seconds):
        if False:
            while True:
                i = 10
        '\n        start_later(seconds) -> None\n\n        Schedule the greenlet to run in the future loop iteration\n        *seconds* later\n        '
        if self._start_event is None:
            _call_spawn_callbacks(self)
            hub = get_my_hub(self)
            self._start_event = hub.loop.timer(seconds)
            self._start_event.start(self.switch)

    @staticmethod
    def add_spawn_callback(callback):
        if False:
            while True:
                i = 10
        '\n        add_spawn_callback(callback) -> None\n\n        Set up a *callback* to be invoked when :class:`Greenlet` objects\n        are started.\n\n        The invocation order of spawn callbacks is unspecified.  Adding the\n        same callback more than one time will not cause it to be called more\n        than once.\n\n        .. versionadded:: 1.4.0\n        '
        global _spawn_callbacks
        if _spawn_callbacks is None:
            _spawn_callbacks = set()
        _spawn_callbacks.add(callback)

    @staticmethod
    def remove_spawn_callback(callback):
        if False:
            return 10
        '\n        remove_spawn_callback(callback) -> None\n\n        Remove *callback* function added with :meth:`Greenlet.add_spawn_callback`.\n        This function will not fail if *callback* has been already removed or\n        if *callback* was never added.\n\n        .. versionadded:: 1.4.0\n        '
        global _spawn_callbacks
        if _spawn_callbacks is not None:
            _spawn_callbacks.discard(callback)
            if not _spawn_callbacks:
                _spawn_callbacks = None

    @classmethod
    def spawn(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        spawn(function, *args, **kwargs) -> Greenlet\n\n        Create a new :class:`Greenlet` object and schedule it to run ``function(*args, **kwargs)``.\n        This can be used as ``gevent.spawn`` or ``Greenlet.spawn``.\n\n        The arguments are passed to :meth:`Greenlet.__init__`.\n\n        .. versionchanged:: 1.1b1\n            If a *function* is given that is not callable, immediately raise a :exc:`TypeError`\n            instead of spawning a greenlet that will raise an uncaught TypeError.\n        '
        g = cls(*args, **kwargs)
        g.start()
        return g

    @classmethod
    def spawn_later(cls, seconds, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        spawn_later(seconds, function, *args, **kwargs) -> Greenlet\n\n        Create and return a new `Greenlet` object scheduled to run ``function(*args, **kwargs)``\n        in a future loop iteration *seconds* later. This can be used as ``Greenlet.spawn_later``\n        or ``gevent.spawn_later``.\n\n        The arguments are passed to :meth:`Greenlet.__init__`.\n\n        .. versionchanged:: 1.1b1\n           If an argument that's meant to be a function (the first argument in *args*, or the ``run`` keyword )\n           is given to this classmethod (and not a classmethod of a subclass),\n           it is verified to be callable. Previously, the spawned greenlet would have failed\n           when it started running.\n        "
        if cls is Greenlet and (not args) and ('run' not in kwargs):
            raise TypeError('')
        g = cls(*args, **kwargs)
        g.start_later(seconds)
        return g

    def _maybe_kill_before_start(self, exception):
        if False:
            i = 10
            return i + 15
        self.__cancel_start()
        self.__free()
        dead = self.dead
        if dead:
            if isinstance(exception, tuple) and len(exception) == 3:
                args = exception
            else:
                args = (exception,)
            self.__handle_death_before_start(args)
        return dead

    def kill(self, exception=GreenletExit, block=True, timeout=None):
        if False:
            print('Hello World!')
        '\n        Raise the ``exception`` in the greenlet.\n\n        If ``block`` is ``True`` (the default), wait until the greenlet\n        dies or the optional timeout expires; this may require switching\n        greenlets.\n        If block is ``False``, the current greenlet is not unscheduled.\n\n        This function always returns ``None`` and never raises an error. It\n        may be called multpile times on the same greenlet object, and may be\n        called on an unstarted or dead greenlet.\n\n        .. note::\n\n            Depending on what this greenlet is executing and the state\n            of the event loop, the exception may or may not be raised\n            immediately when this greenlet resumes execution. It may\n            be raised on a subsequent green call, or, if this greenlet\n            exits before making such a call, it may not be raised at\n            all. As of 1.1, an example where the exception is raised\n            later is if this greenlet had called :func:`sleep(0)\n            <gevent.sleep>`; an example where the exception is raised\n            immediately is if this greenlet had called\n            :func:`sleep(0.1) <gevent.sleep>`.\n\n        .. caution::\n\n            Use care when killing greenlets. If the code executing is not\n            exception safe (e.g., makes proper use of ``finally``) then an\n            unexpected exception could result in corrupted state. Using\n            a :meth:`link` or :meth:`rawlink` (cheaper) may be a safer way to\n            clean up resources.\n\n        See also :func:`gevent.kill` and :func:`gevent.killall`.\n\n        :keyword type exception: The type of exception to raise in the greenlet. The default\n            is :class:`GreenletExit`, which indicates a :meth:`successful` completion\n            of the greenlet.\n\n        .. versionchanged:: 0.13.0\n            *block* is now ``True`` by default.\n        .. versionchanged:: 1.1a2\n            If this greenlet had never been switched to, killing it will\n            prevent it from *ever* being switched to. Links (:meth:`rawlink`)\n            will still be executed, though.\n        .. versionchanged:: 20.12.1\n            If this greenlet is :meth:`ready`, immediately return instead of\n            requiring a trip around the event loop.\n        '
        if not self._maybe_kill_before_start(exception):
            if self.ready():
                return
            waiter = Waiter() if block else None
            hub = get_my_hub(self)
            hub.loop.run_callback(_kill, self, exception, waiter)
            if waiter is not None:
                waiter.get()
                self.join(timeout)

    def get(self, block=True, timeout=None):
        if False:
            while True:
                i = 10
        '\n        get(block=True, timeout=None) -> object\n\n        Return the result the greenlet has returned or re-raise the\n        exception it has raised.\n\n        If block is ``False``, raise :class:`gevent.Timeout` if the\n        greenlet is still alive. If block is ``True``, unschedule the\n        current greenlet until the result is available or the timeout\n        expires. In the latter case, :class:`gevent.Timeout` is\n        raised.\n        '
        if self.ready():
            if self.successful():
                return self.value
            self._raise_exception()
        if not block:
            raise Timeout()
        switch = getcurrent().switch
        self.rawlink(switch)
        try:
            t = Timeout._start_new_or_dummy(timeout)
            try:
                result = get_my_hub(self).switch()
                if result is not self:
                    raise InvalidSwitchError('Invalid switch into Greenlet.get(): %r' % (result,))
            finally:
                t.cancel()
        except:
            self.unlink(switch)
            raise
        if self.ready():
            if self.successful():
                return self.value
            self._raise_exception()

    def join(self, timeout=None):
        if False:
            while True:
                i = 10
        '\n        join(timeout=None) -> None\n\n        Wait until the greenlet finishes or *timeout* expires. Return\n        ``None`` regardless.\n        '
        if self.ready():
            return
        switch = getcurrent().switch
        self.rawlink(switch)
        try:
            t = Timeout._start_new_or_dummy(timeout)
            try:
                result = get_my_hub(self).switch()
                if result is not self:
                    raise InvalidSwitchError('Invalid switch into Greenlet.join(): %r' % (result,))
            finally:
                t.cancel()
        except Timeout as ex:
            self.unlink(switch)
            if ex is not t:
                raise
        except:
            self.unlink(switch)
            raise

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, t, v, tb):
        if False:
            while True:
                i = 10
        if t is None:
            try:
                self.join()
            finally:
                self.kill()
        else:
            self.kill((t, v, tb))

    def __report_result(self, result):
        if False:
            print('Hello World!')
        self._exc_info = (None, None, None)
        self.value = result
        if self._links and (not self._notifier):
            hub = get_my_hub(self)
            self._notifier = hub.loop.run_callback(self._notify_links)

    def __report_error(self, exc_info):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(exc_info[1], GreenletExit):
            self.__report_result(exc_info[1])
            return
        try:
            tb = dump_traceback(exc_info[2])
        except:
            tb = None
        self._exc_info = (exc_info[0], exc_info[1], tb)
        hub = get_my_hub(self)
        if self._links and (not self._notifier):
            self._notifier = hub.loop.run_callback(self._notify_links)
        try:
            hub.handle_error(self, *exc_info)
        finally:
            del exc_info

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.__cancel_start()
            self._start_event = _start_completed_event
            try:
                result = self._run(*self.args, **self.kwargs)
            except:
                self.__report_error(sys_exc_info())
            else:
                self.__report_result(result)
        finally:
            self.__free()

    def __free(self):
        if False:
            return 10
        try:
            del self._run
        except AttributeError:
            pass
        self.args = ()
        self.kwargs.clear()

    def _run(self):
        if False:
            return 10
        '\n        Subclasses may override this method to take any number of\n        arguments and keyword arguments.\n\n        .. versionadded:: 1.1a3\n            Previously, if no callable object was\n            passed to the constructor, the spawned greenlet would later\n            fail with an AttributeError.\n        '
        return

    def has_links(self):
        if False:
            while True:
                i = 10
        return len(self._links)

    def rawlink(self, callback):
        if False:
            return 10
        '\n        Register a callable to be executed when the greenlet finishes\n        execution.\n\n        The *callback* will be called with this instance as an\n        argument.\n\n        The *callback* will be called even if linked after the greenlet\n        is already ready().\n\n        .. caution::\n            The *callback* will be called in the hub and\n            **MUST NOT** raise an exception.\n        '
        if not callable(callback):
            raise TypeError('Expected callable: %r' % (callback,))
        self._links.append(callback)
        if self.ready() and self._links and (not self._notifier):
            hub = get_my_hub(self)
            self._notifier = hub.loop.run_callback(self._notify_links)

    def link(self, callback, SpawnedLink=SpawnedLink):
        if False:
            while True:
                i = 10
        "\n        Link greenlet's completion to a callable.\n\n        The *callback* will be called with this instance as an\n        argument once this greenlet is dead. A callable is called in\n        its own :class:`greenlet.greenlet` (*not* a\n        :class:`Greenlet`).\n\n        The *callback* will be called even if linked after the greenlet\n        is already ready().\n        "
        self.rawlink(SpawnedLink(callback))

    def unlink(self, callback):
        if False:
            print('Hello World!')
        'Remove the callback set by :meth:`link` or :meth:`rawlink`'
        try:
            self._links.remove(callback)
        except ValueError:
            pass

    def unlink_all(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove all the callbacks.\n\n        .. versionadded:: 1.3a2\n        '
        del self._links[:]

    def link_value(self, callback, SpawnedLink=SuccessSpawnedLink):
        if False:
            i = 10
            return i + 15
        '\n        Like :meth:`link` but *callback* is only notified when the greenlet\n        has completed successfully.\n        '
        self.link(callback, SpawnedLink=SpawnedLink)

    def link_exception(self, callback, SpawnedLink=FailureSpawnedLink):
        if False:
            while True:
                i = 10
        '\n        Like :meth:`link` but *callback* is only notified when the\n        greenlet dies because of an unhandled exception.\n        '
        self.link(callback, SpawnedLink=SpawnedLink)

    def _notify_links(self):
        if False:
            for i in range(10):
                print('nop')
        while self._links:
            link = self._links.pop(0)
            try:
                link(self)
            except:
                get_my_hub(self).handle_error((link, self), *sys_exc_info())

class _dummy_event(object):
    __slots__ = ('pending', 'active')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.pending = self.active = False

    def stop(self):
        if False:
            print('Hello World!')
        pass

    def start(self, cb):
        if False:
            return 10
        raise AssertionError('Cannot start the dummy event')

    def close(self):
        if False:
            i = 10
            return i + 15
        pass
_cancelled_start_event = _dummy_event()
_start_completed_event = _dummy_event()

def _kill(glet, exception, waiter):
    if False:
        for i in range(10):
            print('nop')
    try:
        if isinstance(exception, tuple) and len(exception) == 3:
            glet.throw(*exception)
        else:
            glet.throw(exception)
    except:
        get_my_hub(glet).handle_error(glet, *sys_exc_info())
    if waiter is not None:
        waiter.switch(None)

def joinall(greenlets, timeout=None, raise_error=False, count=None):
    if False:
        i = 10
        return i + 15
    '\n    Wait for the ``greenlets`` to finish.\n\n    :param greenlets: A sequence (supporting :func:`len`) of greenlets to wait for.\n    :keyword float timeout: If given, the maximum number of seconds to wait.\n    :return: A sequence of the greenlets that finished before the timeout (if any)\n        expired.\n    '
    if not raise_error:
        return wait(greenlets, timeout=timeout, count=count)
    done = []
    for obj in iwait(greenlets, timeout=timeout, count=count):
        if getattr(obj, 'exception', None) is not None:
            if hasattr(obj, '_raise_exception'):
                obj._raise_exception()
            else:
                raise obj.exception
        done.append(obj)
    return done

def _killall3(greenlets, exception, waiter):
    if False:
        while True:
            i = 10
    diehards = []
    for g in greenlets:
        if not g.dead:
            try:
                g.throw(exception)
            except:
                get_my_hub(g).handle_error(g, *sys_exc_info())
            if not g.dead:
                diehards.append(g)
    waiter.switch(diehards)

def _killall(greenlets, exception):
    if False:
        i = 10
        return i + 15
    for g in greenlets:
        if not g.dead:
            try:
                g.throw(exception)
            except:
                get_my_hub(g).handle_error(g, *sys_exc_info())

def _call_spawn_callbacks(gr):
    if False:
        i = 10
        return i + 15
    if _spawn_callbacks is not None:
        for cb in _spawn_callbacks:
            cb(gr)
_spawn_callbacks = None

def killall(greenlets, exception=GreenletExit, block=True, timeout=None):
    if False:
        return 10
    "\n    Forceably terminate all the *greenlets* by causing them to raise *exception*.\n\n    .. caution:: Use care when killing greenlets. If they are not prepared for exceptions,\n       this could result in corrupted state.\n\n    :param greenlets: A **bounded** iterable of the non-None greenlets to terminate.\n       *All* the items in this iterable must be greenlets that belong to the same hub,\n       which should be the hub for this current thread. If this is a generator or iterator\n       that switches greenlets, the results are undefined.\n    :keyword exception: The type of exception to raise in the greenlets. By default this is\n        :class:`GreenletExit`.\n    :keyword bool block: If True (the default) then this function only returns when all the\n        greenlets are dead; the current greenlet is unscheduled during that process.\n        If greenlets ignore the initial exception raised in them,\n        then they will be joined (with :func:`gevent.joinall`) and allowed to die naturally.\n        If False, this function returns immediately and greenlets will raise\n        the exception asynchronously.\n    :keyword float timeout: A time in seconds to wait for greenlets to die. If given, it is\n        only honored when ``block`` is True.\n    :raise Timeout: If blocking and a timeout is given that elapses before\n        all the greenlets are dead.\n\n    .. versionchanged:: 1.1a2\n        *greenlets* can be any iterable of greenlets, like an iterator or a set.\n        Previously it had to be a list or tuple.\n    .. versionchanged:: 1.5a3\n        Any :class:`Greenlet` in the *greenlets* list that hadn't been switched to before\n        calling this method will never be switched to. This makes this function\n        behave like :meth:`Greenlet.kill`. This does not apply to raw greenlets.\n    .. versionchanged:: 1.5a3\n        Now accepts raw greenlets created by :func:`gevent.spawn_raw`.\n    "
    need_killed = []
    for glet in greenlets:
        try:
            cancel = glet._maybe_kill_before_start
        except AttributeError:
            need_killed.append(glet)
        else:
            if not cancel(exception):
                need_killed.append(glet)
    if not need_killed:
        return
    loop = glet.loop
    if block:
        waiter = Waiter()
        loop.run_callback(_killall3, need_killed, exception, waiter)
        t = Timeout._start_new_or_dummy(timeout)
        try:
            alive = waiter.get()
            if alive:
                joinall(alive, raise_error=False)
        finally:
            t.cancel()
    else:
        loop.run_callback(_killall, need_killed, exception)

def _init():
    if False:
        i = 10
        return i + 15
    greenlet_init()
_init()
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent._greenlet')