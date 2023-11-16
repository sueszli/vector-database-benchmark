"""
Automata with states, transitions and actions.

TODO:
    - add documentation for ioevent, as_supersocket...
"""
import ctypes
import itertools
import logging
import os
import random
import socket
import sys
import threading
import time
import traceback
import types
import select
from collections import deque
from scapy.config import conf
from scapy.utils import do_graph
from scapy.error import log_runtime, warning
from scapy.plist import PacketList
from scapy.data import MTU
from scapy.supersocket import SuperSocket
from scapy.packet import Packet
from scapy.consts import WINDOWS
from typing import Any, Callable, Deque, Dict, Generic, Iterable, Iterator, List, Optional, Set, Tuple, Type, TypeVar, Union, cast
from scapy.compat import DecoratorCallable

def select_objects(inputs, remain):
    if False:
        for i in range(10):
            print('nop')
    '\n    Select objects. Same than:\n    ``select.select(inputs, [], [], remain)``\n\n    But also works on Windows, only on objects whose fileno() returns\n    a Windows event. For simplicity, just use `ObjectPipe()` as a queue\n    that you can select on whatever the platform is.\n\n    If you want an object to be always included in the output of\n    select_objects (i.e. it\'s not selectable), just make fileno()\n    return a strictly negative value.\n\n    Example:\n\n        >>> a, b = ObjectPipe("a"), ObjectPipe("b")\n        >>> b.send("test")\n        >>> select_objects([a, b], 1)\n        [b]\n\n    :param inputs: objects to process\n    :param remain: timeout. If 0, poll.\n    '
    if not WINDOWS:
        return select.select(inputs, [], [], remain)[0]
    natives = []
    events = []
    results = set()
    for i in list(inputs):
        if getattr(i, '__selectable_force_select__', False):
            natives.append(i)
        elif i.fileno() < 0:
            results.add(i)
        else:
            events.append(i)
    if natives:
        results = results.union(set(select.select(natives, [], [], remain)[0]))
    if events:
        remainms = int(remain * 1000 if remain is not None else 4294967295)
        if len(events) == 1:
            res = ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(events[0].fileno()), remainms)
        else:
            res = ctypes.windll.kernel32.WaitForMultipleObjects(len(events), (ctypes.c_void_p * len(events))(*[x.fileno() for x in events]), False, remainms)
        if res != 4294967295 and res != 258:
            results.add(events[res])
            if len(events) > 1:
                for evt in events:
                    res = ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(evt.fileno()), 0)
                    if res == 0:
                        results.add(evt)
    return list(results)
_T = TypeVar('_T')

class ObjectPipe(Generic[_T]):

    def __init__(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        self.name = name or 'ObjectPipe'
        self.closed = False
        (self.__rd, self.__wr) = os.pipe()
        self.__queue = deque()
        if WINDOWS:
            self._wincreate()
    if WINDOWS:

        def _wincreate(self):
            if False:
                print('Hello World!')
            self._fd = cast(int, ctypes.windll.kernel32.CreateEventA(None, True, False, ctypes.create_string_buffer(b'ObjectPipe %f' % random.random())))

        def _winset(self):
            if False:
                while True:
                    i = 10
            if ctypes.windll.kernel32.SetEvent(ctypes.c_void_p(self._fd)) == 0:
                warning(ctypes.FormatError(ctypes.GetLastError()))

        def _winreset(self):
            if False:
                i = 10
                return i + 15
            if ctypes.windll.kernel32.ResetEvent(ctypes.c_void_p(self._fd)) == 0:
                warning(ctypes.FormatError(ctypes.GetLastError()))

        def _winclose(self):
            if False:
                while True:
                    i = 10
            if ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._fd)) == 0:
                warning(ctypes.FormatError(ctypes.GetLastError()))

    def fileno(self):
        if False:
            while True:
                i = 10
        if WINDOWS:
            return self._fd
        return self.__rd

    def send(self, obj):
        if False:
            i = 10
            return i + 15
        self.__queue.append(obj)
        if WINDOWS:
            self._winset()
        os.write(self.__wr, b'X')
        return 1

    def write(self, obj):
        if False:
            while True:
                i = 10
        self.send(obj)

    def empty(self):
        if False:
            print('Hello World!')
        return not bool(self.__queue)

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def recv(self, n=0):
        if False:
            print('Hello World!')
        if self.closed:
            if self.__queue:
                return self.__queue.popleft()
            return None
        os.read(self.__rd, 1)
        elt = self.__queue.popleft()
        if WINDOWS and (not self.__queue):
            self._winreset()
        return elt

    def read(self, n=0):
        if False:
            return 10
        return self.recv(n)

    def clear(self):
        if False:
            while True:
                i = 10
        if not self.closed:
            while not self.empty():
                self.recv()

    def close(self):
        if False:
            i = 10
            return i + 15
        if not self.closed:
            os.close(self.__rd)
            os.close(self.__wr)
            if WINDOWS:
                self._winclose()
            self.closed = True

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s at %s>' % (self.name, id(self))

    def __del__(self):
        if False:
            print('Hello World!')
        self.close()

    @staticmethod
    def select(sockets, remain=conf.recv_poll_rate):
        if False:
            print('Hello World!')
        results = []
        for s in sockets:
            if s.closed:
                results.append(s)
        if results:
            return results
        return select_objects(sockets, remain)

class Message:
    type = None
    pkt = None
    result = None
    state = None
    exc_info = None

    def __init__(self, **args):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(args)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<Message %s>' % ' '.join(('%s=%r' % (k, v) for (k, v) in self.__dict__.items() if not k.startswith('_')))

class Timer:

    def __init__(self, time, prio=0, autoreload=False):
        if False:
            return 10
        self._timeout = float(time)
        self._time = 0
        self._just_expired = True
        self._expired = True
        self._prio = prio
        self._func = _StateWrapper()
        self._autoreload = autoreload

    def get(self):
        if False:
            print('Hello World!')
        return self._timeout

    def set(self, val):
        if False:
            print('Hello World!')
        self._timeout = val

    def _reset(self):
        if False:
            while True:
                i = 10
        self._time = self._timeout
        self._expired = False
        self._just_expired = False

    def _reset_just_expired(self):
        if False:
            print('Hello World!')
        self._just_expired = False

    def _running(self):
        if False:
            i = 10
            return i + 15
        return self._time > 0

    def _remaining(self):
        if False:
            i = 10
            return i + 15
        return max(self._time, 0)

    def _decrement(self, time):
        if False:
            for i in range(10):
                print('nop')
        self._time -= time
        if self._time <= 0:
            if not self._expired:
                self._just_expired = True
                if self._autoreload:
                    self._time = self._timeout + self._time
                else:
                    self._expired = True
                    self._time = 0

    def __lt__(self, obj):
        if False:
            while True:
                i = 10
        return self._time < obj._time if self._time != obj._time else self._prio < obj._prio

    def __gt__(self, obj):
        if False:
            print('Hello World!')
        return self._time > obj._time if self._time != obj._time else self._prio > obj._prio

    def __eq__(self, obj):
        if False:
            return 10
        if not isinstance(obj, Timer):
            raise NotImplementedError()
        return self._time == obj._time and self._prio == obj._prio

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<Timer %f(%f)>' % (self._time, self._timeout)

class _TimerList:

    def __init__(self):
        if False:
            return 10
        self.timers = []

    def add_timer(self, timer):
        if False:
            while True:
                i = 10
        self.timers.append(timer)

    def reset(self):
        if False:
            print('Hello World!')
        for t in self.timers:
            t._reset()

    def decrement(self, time):
        if False:
            while True:
                i = 10
        for t in self.timers:
            t._decrement(time)

    def expired(self):
        if False:
            return 10
        lst = [t for t in self.timers if t._just_expired]
        lst.sort(key=lambda x: x._prio, reverse=True)
        for t in lst:
            t._reset_just_expired()
        return lst

    def until_next(self):
        if False:
            while True:
                i = 10
        try:
            return min([t._remaining() for t in self.timers if t._running()])
        except ValueError:
            return 0

    def count(self):
        if False:
            while True:
                i = 10
        return len(self.timers)

    def __iter__(self):
        if False:
            return 10
        return self.timers.__iter__()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.timers.__repr__()

class _instance_state:

    def __init__(self, instance):
        if False:
            while True:
                i = 10
        self.__self__ = instance.__self__
        self.__func__ = instance.__func__
        self.__self__.__class__ = instance.__self__.__class__

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        return getattr(self.__func__, attr)

    def __call__(self, *args, **kargs):
        if False:
            print('Hello World!')
        return self.__func__(self.__self__, *args, **kargs)

    def breaks(self):
        if False:
            i = 10
            return i + 15
        return self.__self__.add_breakpoints(self.__func__)

    def intercepts(self):
        if False:
            i = 10
            return i + 15
        return self.__self__.add_interception_points(self.__func__)

    def unbreaks(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__self__.remove_breakpoints(self.__func__)

    def unintercepts(self):
        if False:
            print('Hello World!')
        return self.__self__.remove_interception_points(self.__func__)

class _StateWrapper:
    __name__ = None
    atmt_type = None
    atmt_state = None
    atmt_initial = None
    atmt_final = None
    atmt_stop = None
    atmt_error = None
    atmt_origfunc = None
    atmt_prio = None
    atmt_as_supersocket = None
    atmt_condname = None
    atmt_ioname = None
    atmt_timeout = None
    atmt_cond = None
    __code__ = None
    __call__ = None

class ATMT:
    STATE = 'State'
    ACTION = 'Action'
    CONDITION = 'Condition'
    RECV = 'Receive condition'
    TIMEOUT = 'Timeout condition'
    IOEVENT = 'I/O event'

    class NewStateRequested(Exception):

        def __init__(self, state_func, automaton, *args, **kargs):
            if False:
                i = 10
                return i + 15
            self.func = state_func
            self.state = state_func.atmt_state
            self.initial = state_func.atmt_initial
            self.error = state_func.atmt_error
            self.stop = state_func.atmt_stop
            self.final = state_func.atmt_final
            Exception.__init__(self, 'Request state [%s]' % self.state)
            self.automaton = automaton
            self.args = args
            self.kargs = kargs
            self.action_parameters()

        def action_parameters(self, *args, **kargs):
            if False:
                while True:
                    i = 10
            self.action_args = args
            self.action_kargs = kargs
            return self

        def run(self):
            if False:
                return 10
            return self.func(self.automaton, *self.args, **self.kargs)

        def __repr__(self):
            if False:
                while True:
                    i = 10
            return 'NewStateRequested(%s)' % self.state

    @staticmethod
    def state(initial=0, final=0, stop=0, error=0):
        if False:
            i = 10
            return i + 15

        def deco(f, initial=initial, final=final):
            if False:
                while True:
                    i = 10
            f.atmt_type = ATMT.STATE
            f.atmt_state = f.__name__
            f.atmt_initial = initial
            f.atmt_final = final
            f.atmt_stop = stop
            f.atmt_error = error

            def _state_wrapper(self, *args, **kargs):
                if False:
                    print('Hello World!')
                return ATMT.NewStateRequested(f, self, *args, **kargs)
            state_wrapper = cast(_StateWrapper, _state_wrapper)
            state_wrapper.__name__ = '%s_wrapper' % f.__name__
            state_wrapper.atmt_type = ATMT.STATE
            state_wrapper.atmt_state = f.__name__
            state_wrapper.atmt_initial = initial
            state_wrapper.atmt_final = final
            state_wrapper.atmt_stop = stop
            state_wrapper.atmt_error = error
            state_wrapper.atmt_origfunc = f
            return state_wrapper
        return deco

    @staticmethod
    def action(cond, prio=0):
        if False:
            for i in range(10):
                print('nop')

        def deco(f, cond=cond):
            if False:
                i = 10
                return i + 15
            if not hasattr(f, 'atmt_type'):
                f.atmt_cond = {}
            f.atmt_type = ATMT.ACTION
            f.atmt_cond[cond.atmt_condname] = prio
            return f
        return deco

    @staticmethod
    def condition(state, prio=0):
        if False:
            while True:
                i = 10

        def deco(f, state=state):
            if False:
                print('Hello World!')
            f.atmt_type = ATMT.CONDITION
            f.atmt_state = state.atmt_state
            f.atmt_condname = f.__name__
            f.atmt_prio = prio
            return f
        return deco

    @staticmethod
    def receive_condition(state, prio=0):
        if False:
            i = 10
            return i + 15

        def deco(f, state=state):
            if False:
                print('Hello World!')
            f.atmt_type = ATMT.RECV
            f.atmt_state = state.atmt_state
            f.atmt_condname = f.__name__
            f.atmt_prio = prio
            return f
        return deco

    @staticmethod
    def ioevent(state, name, prio=0, as_supersocket=None):
        if False:
            print('Hello World!')

        def deco(f, state=state):
            if False:
                print('Hello World!')
            f.atmt_type = ATMT.IOEVENT
            f.atmt_state = state.atmt_state
            f.atmt_condname = f.__name__
            f.atmt_ioname = name
            f.atmt_prio = prio
            f.atmt_as_supersocket = as_supersocket
            return f
        return deco

    @staticmethod
    def timeout(state, timeout):
        if False:
            for i in range(10):
                print('nop')

        def deco(f, state=state, timeout=Timer(timeout)):
            if False:
                return 10
            f.atmt_type = ATMT.TIMEOUT
            f.atmt_state = state.atmt_state
            f.atmt_timeout = timeout
            f.atmt_timeout._func = f
            f.atmt_condname = f.__name__
            return f
        return deco

    @staticmethod
    def timer(state, timeout, prio=0):
        if False:
            return 10

        def deco(f, state=state, timeout=Timer(timeout, prio=prio, autoreload=True)):
            if False:
                return 10
            f.atmt_type = ATMT.TIMEOUT
            f.atmt_state = state.atmt_state
            f.atmt_timeout = timeout
            f.atmt_timeout._func = f
            f.atmt_condname = f.__name__
            return f
        return deco

class _ATMT_Command:
    RUN = 'RUN'
    NEXT = 'NEXT'
    FREEZE = 'FREEZE'
    STOP = 'STOP'
    FORCESTOP = 'FORCESTOP'
    END = 'END'
    EXCEPTION = 'EXCEPTION'
    SINGLESTEP = 'SINGLESTEP'
    BREAKPOINT = 'BREAKPOINT'
    INTERCEPT = 'INTERCEPT'
    ACCEPT = 'ACCEPT'
    REPLACE = 'REPLACE'
    REJECT = 'REJECT'

class _ATMT_supersocket(SuperSocket):

    def __init__(self, name, ioevent, automaton, proto, *args, **kargs):
        if False:
            while True:
                i = 10
        self.name = name
        self.ioevent = ioevent
        self.proto = proto
        (self.spa, self.spb) = (ObjectPipe[bytes]('spa'), ObjectPipe[bytes]('spb'))
        kargs['external_fd'] = {ioevent: (self.spa, self.spb)}
        kargs['is_atmt_socket'] = True
        self.atmt = automaton(*args, **kargs)
        self.atmt.runbg()

    def send(self, s):
        if False:
            i = 10
            return i + 15
        if not isinstance(s, bytes):
            s = bytes(s)
        return self.spa.send(s)

    def fileno(self):
        if False:
            while True:
                i = 10
        return self.spb.fileno()

    def recv(self, n=MTU, **kwargs):
        if False:
            return 10
        r = self.spb.recv(n)
        if self.proto is not None and r is not None:
            r = self.proto(r, **kwargs)
        return r

    def close(self):
        if False:
            while True:
                i = 10
        if not self.closed:
            self.atmt.stop()
            self.atmt.destroy()
            self.spa.close()
            self.spb.close()
            self.closed = True

    @staticmethod
    def select(sockets, remain=conf.recv_poll_rate):
        if False:
            while True:
                i = 10
        return select_objects(sockets, remain)

class _ATMT_to_supersocket:

    def __init__(self, name, ioevent, automaton):
        if False:
            print('Hello World!')
        self.name = name
        self.ioevent = ioevent
        self.automaton = automaton

    def __call__(self, proto, *args, **kargs):
        if False:
            i = 10
            return i + 15
        return _ATMT_supersocket(self.name, self.ioevent, self.automaton, proto, *args, **kargs)

class Automaton_metaclass(type):

    def __new__(cls, name, bases, dct):
        if False:
            print('Hello World!')
        cls = super(Automaton_metaclass, cls).__new__(cls, name, bases, dct)
        cls.states = {}
        cls.recv_conditions = {}
        cls.conditions = {}
        cls.ioevents = {}
        cls.timeout = {}
        cls.actions = {}
        cls.initial_states = []
        cls.stop_states = []
        cls.ionames = []
        cls.iosupersockets = []
        members = {}
        classes = [cls]
        while classes:
            c = classes.pop(0)
            classes += list(c.__bases__)
            for (k, v) in c.__dict__.items():
                if k not in members:
                    members[k] = v
        decorated = [v for v in members.values() if hasattr(v, 'atmt_type')]
        for m in decorated:
            if m.atmt_type == ATMT.STATE:
                s = m.atmt_state
                cls.states[s] = m
                cls.recv_conditions[s] = []
                cls.ioevents[s] = []
                cls.conditions[s] = []
                cls.timeout[s] = _TimerList()
                if m.atmt_initial:
                    cls.initial_states.append(m)
                if m.atmt_stop:
                    cls.stop_states.append(m)
            elif m.atmt_type in [ATMT.CONDITION, ATMT.RECV, ATMT.TIMEOUT, ATMT.IOEVENT]:
                cls.actions[m.atmt_condname] = []
        for m in decorated:
            if m.atmt_type == ATMT.CONDITION:
                cls.conditions[m.atmt_state].append(m)
            elif m.atmt_type == ATMT.RECV:
                cls.recv_conditions[m.atmt_state].append(m)
            elif m.atmt_type == ATMT.IOEVENT:
                cls.ioevents[m.atmt_state].append(m)
                cls.ionames.append(m.atmt_ioname)
                if m.atmt_as_supersocket is not None:
                    cls.iosupersockets.append(m)
            elif m.atmt_type == ATMT.TIMEOUT:
                cls.timeout[m.atmt_state].add_timer(m.atmt_timeout)
            elif m.atmt_type == ATMT.ACTION:
                for co in m.atmt_cond:
                    cls.actions[co].append(m)
        for v in itertools.chain(cls.conditions.values(), cls.recv_conditions.values(), cls.ioevents.values()):
            v.sort(key=lambda x: x.atmt_prio)
        for (condname, actlst) in cls.actions.items():
            actlst.sort(key=lambda x: x.atmt_cond[condname])
        for ioev in cls.iosupersockets:
            setattr(cls, ioev.atmt_as_supersocket, _ATMT_to_supersocket(ioev.atmt_as_supersocket, ioev.atmt_ioname, cast(Type['Automaton'], cls)))
        try:
            import inspect
            cls.__signature__ = inspect.signature(cls.parse_args)
        except (ImportError, AttributeError):
            pass
        return cast(Type['Automaton'], cls)

    def build_graph(self):
        if False:
            print('Hello World!')
        s = 'digraph "%s" {\n' % self.__class__.__name__
        se = ''
        for st in self.states.values():
            if st.atmt_initial:
                se = '\t"%s" [ style=filled, fillcolor=blue, shape=box, root=true];\n' % st.atmt_state + se
            elif st.atmt_final:
                se += '\t"%s" [ style=filled, fillcolor=green, shape=octagon ];\n' % st.atmt_state
            elif st.atmt_error:
                se += '\t"%s" [ style=filled, fillcolor=red, shape=octagon ];\n' % st.atmt_state
            elif st.atmt_stop:
                se += '\t"%s" [ style=filled, fillcolor=orange, shape=box, root=true ];\n' % st.atmt_state
        s += se
        for st in self.states.values():
            names = list(st.atmt_origfunc.__code__.co_names + st.atmt_origfunc.__code__.co_consts)
            while names:
                n = names.pop()
                if n in self.states:
                    s += '\t"%s" -> "%s" [ color=green ];\n' % (st.atmt_state, n)
                elif n in self.__dict__:
                    if callable(self.__dict__[n]):
                        names.extend(self.__dict__[n].__code__.co_names)
                        names.extend(self.__dict__[n].__code__.co_consts)
        for (c, k, v) in [('purple', k, v) for (k, v) in self.conditions.items()] + [('red', k, v) for (k, v) in self.recv_conditions.items()] + [('orange', k, v) for (k, v) in self.ioevents.items()]:
            for f in v:
                names = list(f.__code__.co_names + f.__code__.co_consts)
                while names:
                    n = names.pop()
                    if n in self.states:
                        line = f.atmt_condname
                        for x in self.actions[f.atmt_condname]:
                            line += '\\l>[%s]' % x.__name__
                        s += '\t"%s" -> "%s" [label="%s", color=%s];\n' % (k, n, line, c)
                    elif n in self.__dict__:
                        if callable(self.__dict__[n]):
                            names.extend(self.__dict__[n].__code__.co_names)
                            names.extend(self.__dict__[n].__code__.co_consts)
        for (k, timers) in self.timeout.items():
            for timer in timers:
                for n in timer._func.__code__.co_names + timer._func.__code__.co_consts:
                    if n in self.states:
                        line = '%s/%.1fs' % (timer._func.atmt_condname, timer.get())
                        for x in self.actions[timer._func.atmt_condname]:
                            line += '\\l>[%s]' % x.__name__
                        s += '\t"%s" -> "%s" [label="%s",color=blue];\n' % (k, n, line)
        s += '}\n'
        return s

    def graph(self, **kargs):
        if False:
            return 10
        s = self.build_graph()
        return do_graph(s, **kargs)

class Automaton(metaclass=Automaton_metaclass):
    states = {}
    state = None
    recv_conditions = {}
    conditions = {}
    ioevents = {}
    timeout = {}
    actions = {}
    initial_states = []
    stop_states = []
    ionames = []
    iosupersockets = []

    def __init__(self, *args, **kargs):
        if False:
            return 10
        external_fd = kargs.pop('external_fd', {})
        self.send_sock_class = kargs.pop('ll', conf.L3socket)
        self.recv_sock_class = kargs.pop('recvsock', conf.L2listen)
        self.is_atmt_socket = kargs.pop('is_atmt_socket', False)
        self.started = threading.Lock()
        self.threadid = None
        self.breakpointed = None
        self.breakpoints = set()
        self.interception_points = set()
        self.intercepted_packet = None
        self.debug_level = 0
        self.init_args = args
        self.init_kargs = kargs
        self.io = type.__new__(type, 'IOnamespace', (), {})
        self.oi = type.__new__(type, 'IOnamespace', (), {})
        self.cmdin = ObjectPipe[Message]('cmdin')
        self.cmdout = ObjectPipe[Message]('cmdout')
        self.ioin = {}
        self.ioout = {}
        self.packets = PacketList()
        for n in self.__class__.ionames:
            extfd = external_fd.get(n)
            if not isinstance(extfd, tuple):
                extfd = (extfd, extfd)
            (ioin, ioout) = extfd
            if ioin is None:
                ioin = ObjectPipe('ioin')
            else:
                ioin = self._IO_fdwrapper(ioin, None)
            if ioout is None:
                ioout = ObjectPipe('ioout')
            else:
                ioout = self._IO_fdwrapper(None, ioout)
            self.ioin[n] = ioin
            self.ioout[n] = ioout
            ioin.ioname = n
            ioout.ioname = n
            setattr(self.io, n, self._IO_mixer(ioout, ioin))
            setattr(self.oi, n, self._IO_mixer(ioin, ioout))
        for stname in self.states:
            setattr(self, stname, _instance_state(getattr(self, stname)))
        self.start()

    def parse_args(self, debug=0, store=1, **kargs):
        if False:
            print('Hello World!')
        self.debug_level = debug
        if debug:
            conf.logLevel = logging.DEBUG
        self.socket_kargs = kargs
        self.store_packets = store

    def master_filter(self, pkt):
        if False:
            while True:
                i = 10
        return True

    def my_send(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        self.send_sock.send(pkt)

    def timer_by_name(self, name):
        if False:
            while True:
                i = 10
        for (_, timers) in self.timeout.items():
            for timer in timers:
                if timer._func.atmt_condname == name:
                    return timer
        return None

    class _IO_fdwrapper:

        def __init__(self, rd, wr):
            if False:
                return 10
            self.rd = rd
            self.wr = wr
            if isinstance(self.rd, socket.socket):
                self.__selectable_force_select__ = True

        def fileno(self):
            if False:
                i = 10
                return i + 15
            if isinstance(self.rd, int):
                return self.rd
            elif self.rd:
                return self.rd.fileno()
            return 0

        def read(self, n=65535):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(self.rd, int):
                return os.read(self.rd, n)
            elif self.rd:
                return self.rd.recv(n)
            return None

        def write(self, msg):
            if False:
                print('Hello World!')
            if isinstance(self.wr, int):
                return os.write(self.wr, msg)
            elif self.wr:
                return self.wr.send(msg)
            return 0

        def recv(self, n=65535):
            if False:
                print('Hello World!')
            return self.read(n)

        def send(self, msg):
            if False:
                print('Hello World!')
            return self.write(msg)

    class _IO_mixer:

        def __init__(self, rd, wr):
            if False:
                while True:
                    i = 10
            self.rd = rd
            self.wr = wr

        def fileno(self):
            if False:
                return 10
            if isinstance(self.rd, ObjectPipe):
                return self.rd.fileno()
            return self.rd

        def recv(self, n=None):
            if False:
                while True:
                    i = 10
            return self.rd.recv(n)

        def read(self, n=None):
            if False:
                return 10
            return self.recv(n)

        def send(self, msg):
            if False:
                i = 10
                return i + 15
            return self.wr.send(msg)

        def write(self, msg):
            if False:
                return 10
            return self.send(msg)

    class AutomatonException(Exception):

        def __init__(self, msg, state=None, result=None):
            if False:
                return 10
            Exception.__init__(self, msg)
            self.state = state
            self.result = result

    class AutomatonError(AutomatonException):
        pass

    class ErrorState(AutomatonException):
        pass

    class Stuck(AutomatonException):
        pass

    class AutomatonStopped(AutomatonException):
        pass

    class Breakpoint(AutomatonStopped):
        pass

    class Singlestep(AutomatonStopped):
        pass

    class InterceptionPoint(AutomatonStopped):

        def __init__(self, msg, state=None, result=None, packet=None):
            if False:
                while True:
                    i = 10
            Automaton.AutomatonStopped.__init__(self, msg, state=state, result=result)
            self.packet = packet

    class CommandMessage(AutomatonException):
        pass

    def debug(self, lvl, msg):
        if False:
            return 10
        if self.debug_level >= lvl:
            log_runtime.debug(msg)

    def send(self, pkt):
        if False:
            i = 10
            return i + 15
        if self.state.state in self.interception_points:
            self.debug(3, 'INTERCEPT: packet intercepted: %s' % pkt.summary())
            self.intercepted_packet = pkt
            self.cmdout.send(Message(type=_ATMT_Command.INTERCEPT, state=self.state, pkt=pkt))
            cmd = self.cmdin.recv()
            if not cmd:
                self.debug(3, 'CANCELLED')
                return
            self.intercepted_packet = None
            if cmd.type == _ATMT_Command.REJECT:
                self.debug(3, 'INTERCEPT: packet rejected')
                return
            elif cmd.type == _ATMT_Command.REPLACE:
                pkt = cmd.pkt
                self.debug(3, 'INTERCEPT: packet replaced by: %s' % pkt.summary())
            elif cmd.type == _ATMT_Command.ACCEPT:
                self.debug(3, 'INTERCEPT: packet accepted')
            else:
                raise self.AutomatonError('INTERCEPT: unknown verdict: %r' % cmd.type)
        self.my_send(pkt)
        self.debug(3, 'SENT : %s' % pkt.summary())
        if self.store_packets:
            self.packets.append(pkt.copy())

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __del__(self):
        if False:
            print('Hello World!')
        self.stop()
        self.destroy()

    def _run_condition(self, cond, *args, **kargs):
        if False:
            while True:
                i = 10
        try:
            self.debug(5, 'Trying %s [%s]' % (cond.atmt_type, cond.atmt_condname))
            cond(self, *args, **kargs)
        except ATMT.NewStateRequested as state_req:
            self.debug(2, '%s [%s] taken to state [%s]' % (cond.atmt_type, cond.atmt_condname, state_req.state))
            if cond.atmt_type == ATMT.RECV:
                if self.store_packets:
                    self.packets.append(args[0])
            for action in self.actions[cond.atmt_condname]:
                self.debug(2, '   + Running action [%s]' % action.__name__)
                action(self, *state_req.action_args, **state_req.action_kargs)
            raise
        except Exception as e:
            self.debug(2, '%s [%s] raised exception [%s]' % (cond.atmt_type, cond.atmt_condname, e))
            raise
        else:
            self.debug(2, '%s [%s] not taken' % (cond.atmt_type, cond.atmt_condname))

    def _do_start(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        ready = threading.Event()
        _t = threading.Thread(target=self._do_control, args=(ready,) + args, kwargs=kargs, name='scapy.automaton _do_start')
        _t.daemon = True
        _t.start()
        ready.wait()

    def _do_control(self, ready, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        with self.started:
            self.threadid = threading.current_thread().ident
            if self.threadid is None:
                self.threadid = 0
            a = args + self.init_args[len(args):]
            k = self.init_kargs.copy()
            k.update(kargs)
            self.parse_args(*a, **k)
            self.state = self.initial_states[0](self)
            self.send_sock = self.send_sock_class(**self.socket_kargs)
            if self.recv_conditions:
                self.listen_sock = self.recv_sock_class(**self.socket_kargs)
            else:
                self.listen_sock = None
            self.packets = PacketList(name='session[%s]' % self.__class__.__name__)
            singlestep = True
            iterator = self._do_iter()
            self.debug(3, 'Starting control thread [tid=%i]' % self.threadid)
            ready.set()
            try:
                while True:
                    c = self.cmdin.recv()
                    if c is None:
                        return None
                    self.debug(5, 'Received command %s' % c.type)
                    if c.type == _ATMT_Command.RUN:
                        singlestep = False
                    elif c.type == _ATMT_Command.NEXT:
                        singlestep = True
                    elif c.type == _ATMT_Command.FREEZE:
                        continue
                    elif c.type == _ATMT_Command.STOP:
                        if self.stop_states:
                            self.state = self.stop_states[0](self)
                            iterator = self._do_iter()
                        else:
                            break
                    elif c.type == _ATMT_Command.FORCESTOP:
                        break
                    while True:
                        state = next(iterator)
                        if isinstance(state, self.CommandMessage):
                            break
                        elif isinstance(state, self.Breakpoint):
                            c = Message(type=_ATMT_Command.BREAKPOINT, state=state)
                            self.cmdout.send(c)
                            break
                        if singlestep:
                            c = Message(type=_ATMT_Command.SINGLESTEP, state=state)
                            self.cmdout.send(c)
                            break
            except (StopIteration, RuntimeError):
                c = Message(type=_ATMT_Command.END, result=self.final_state_output)
                self.cmdout.send(c)
            except Exception as e:
                exc_info = sys.exc_info()
                self.debug(3, 'Transferring exception from tid=%i:\n%s' % (self.threadid, ''.join(traceback.format_exception(*exc_info))))
                m = Message(type=_ATMT_Command.EXCEPTION, exception=e, exc_info=exc_info)
                self.cmdout.send(m)
            self.debug(3, 'Stopping control thread (tid=%i)' % self.threadid)
            self.threadid = None
            if getattr(self, 'listen_sock', None):
                self.listen_sock.close()
            if getattr(self, 'send_sock', None):
                self.send_sock.close()

    def _do_iter(self):
        if False:
            while True:
                i = 10
        while True:
            try:
                self.debug(1, '## state=[%s]' % self.state.state)
                if self.state.state in self.breakpoints and self.state.state != self.breakpointed:
                    self.breakpointed = self.state.state
                    yield self.Breakpoint('breakpoint triggered on state %s' % self.state.state, state=self.state.state)
                self.breakpointed = None
                state_output = self.state.run()
                if self.state.error:
                    raise self.ErrorState('Reached %s: [%r]' % (self.state.state, state_output), result=state_output, state=self.state.state)
                if self.state.final:
                    self.final_state_output = state_output
                    return
                if state_output is None:
                    state_output = ()
                elif not isinstance(state_output, list):
                    state_output = (state_output,)
                timers = self.timeout[self.state.state]
                if not select_objects([self.cmdin], 0):
                    for cond in self.conditions[self.state.state]:
                        self._run_condition(cond, *state_output)
                    if len(self.recv_conditions[self.state.state]) == 0 and len(self.ioevents[self.state.state]) == 0 and (timers.count() == 0):
                        raise self.Stuck('stuck in [%s]' % self.state.state, state=self.state.state, result=state_output)
                timers.reset()
                time_previous = time.time()
                fds = [self.cmdin]
                if self.listen_sock and self.recv_conditions[self.state.state]:
                    fds.append(self.listen_sock)
                for ioev in self.ioevents[self.state.state]:
                    fds.append(self.ioin[ioev.atmt_ioname])
                while True:
                    time_current = time.time()
                    timers.decrement(time_current - time_previous)
                    time_previous = time_current
                    for timer in timers.expired():
                        self._run_condition(timer._func, *state_output)
                    remain = timers.until_next()
                    self.debug(5, 'Select on %r' % fds)
                    r = select_objects(fds, remain)
                    self.debug(5, 'Selected %r' % r)
                    for fd in r:
                        self.debug(5, 'Looking at %r' % fd)
                        if fd == self.cmdin:
                            yield self.CommandMessage('Received command message')
                        elif fd == self.listen_sock:
                            pkt = self.listen_sock.recv(MTU)
                            if pkt is not None:
                                if self.master_filter(pkt):
                                    self.debug(3, 'RECVD: %s' % pkt.summary())
                                    for rcvcond in self.recv_conditions[self.state.state]:
                                        self._run_condition(rcvcond, pkt, *state_output)
                                else:
                                    self.debug(4, 'FILTR: %s' % pkt.summary())
                        else:
                            self.debug(3, 'IOEVENT on %s' % fd.ioname)
                            for ioevt in self.ioevents[self.state.state]:
                                if ioevt.atmt_ioname == fd.ioname:
                                    self._run_condition(ioevt, fd, *state_output)
            except ATMT.NewStateRequested as state_req:
                self.debug(2, 'switching from [%s] to [%s]' % (self.state.state, state_req.state))
                self.state = state_req
                yield state_req

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<Automaton %s [%s]>' % (self.__class__.__name__, ['HALTED', 'RUNNING'][self.started.locked()])

    def add_interception_points(self, *ipts):
        if False:
            for i in range(10):
                print('nop')
        for ipt in ipts:
            if hasattr(ipt, 'atmt_state'):
                ipt = ipt.atmt_state
            self.interception_points.add(ipt)

    def remove_interception_points(self, *ipts):
        if False:
            print('Hello World!')
        for ipt in ipts:
            if hasattr(ipt, 'atmt_state'):
                ipt = ipt.atmt_state
            self.interception_points.discard(ipt)

    def add_breakpoints(self, *bps):
        if False:
            for i in range(10):
                print('nop')
        for bp in bps:
            if hasattr(bp, 'atmt_state'):
                bp = bp.atmt_state
            self.breakpoints.add(bp)

    def remove_breakpoints(self, *bps):
        if False:
            print('Hello World!')
        for bp in bps:
            if hasattr(bp, 'atmt_state'):
                bp = bp.atmt_state
            self.breakpoints.discard(bp)

    def start(self, *args, **kargs):
        if False:
            while True:
                i = 10
        if self.started.locked():
            raise ValueError('Already started')
        self._do_start(*args, **kargs)

    def run(self, resume=None, wait=True):
        if False:
            i = 10
            return i + 15
        if resume is None:
            resume = Message(type=_ATMT_Command.RUN)
        self.cmdin.send(resume)
        if wait:
            try:
                c = self.cmdout.recv()
                if c is None:
                    return None
            except KeyboardInterrupt:
                self.cmdin.send(Message(type=_ATMT_Command.FREEZE))
                return None
            if c.type == _ATMT_Command.END:
                return c.result
            elif c.type == _ATMT_Command.INTERCEPT:
                raise self.InterceptionPoint('packet intercepted', state=c.state.state, packet=c.pkt)
            elif c.type == _ATMT_Command.SINGLESTEP:
                raise self.Singlestep('singlestep state=[%s]' % c.state.state, state=c.state.state)
            elif c.type == _ATMT_Command.BREAKPOINT:
                raise self.Breakpoint('breakpoint triggered on state [%s]' % c.state.state, state=c.state.state)
            elif c.type == _ATMT_Command.EXCEPTION:
                value = c.exc_info[0]() if c.exc_info[1] is None else c.exc_info[1]
                if value.__traceback__ is not c.exc_info[2]:
                    raise value.with_traceback(c.exc_info[2])
                raise value
        return None

    def runbg(self, resume=None, wait=False):
        if False:
            i = 10
            return i + 15
        self.run(resume, wait)

    def __next__(self):
        if False:
            i = 10
            return i + 15
        return self.run(resume=Message(type=_ATMT_Command.NEXT))

    def _flush_inout(self):
        if False:
            return 10
        for cmd in [self.cmdin, self.cmdout]:
            cmd.clear()

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Destroys a stopped Automaton: this cleanups all opened file descriptors.\n        Required on PyPy for instance where the garbage collector behaves differently.\n        '
        if self.started.locked():
            raise ValueError("Can't close running Automaton ! Call stop() beforehand")
        self._flush_inout()
        self.cmdin.close()
        self.cmdout.close()
        for i in itertools.chain(self.ioin.values(), self.ioout.values()):
            if isinstance(i, ObjectPipe):
                i.close()

    def stop(self, wait=True):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.cmdin.send(Message(type=_ATMT_Command.STOP))
        except OSError:
            pass
        if wait:
            with self.started:
                self._flush_inout()

    def forcestop(self, wait=True):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.cmdin.send(Message(type=_ATMT_Command.FORCESTOP))
        except OSError:
            pass
        if wait:
            with self.started:
                self._flush_inout()

    def restart(self, *args, **kargs):
        if False:
            while True:
                i = 10
        self.stop()
        self.start(*args, **kargs)

    def accept_packet(self, pkt=None, wait=False):
        if False:
            i = 10
            return i + 15
        rsm = Message()
        if pkt is None:
            rsm.type = _ATMT_Command.ACCEPT
        else:
            rsm.type = _ATMT_Command.REPLACE
            rsm.pkt = pkt
        return self.run(resume=rsm, wait=wait)

    def reject_packet(self, wait=False):
        if False:
            return 10
        rsm = Message(type=_ATMT_Command.REJECT)
        return self.run(resume=rsm, wait=wait)