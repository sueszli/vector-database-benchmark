"""Classes for running 0MQ Devices in the background."""
import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device

class Device:
    """A 0MQ Device to be run in the background.

    You do not pass Socket instances to this, but rather Socket types::

        Device(device_type, in_socket_type, out_socket_type)

    For instance::

        dev = Device(zmq.QUEUE, zmq.DEALER, zmq.ROUTER)

    Similar to zmq.device, but socket types instead of sockets themselves are
    passed, and the sockets are created in the work thread, to avoid issues
    with thread safety. As a result, additional bind_{in|out} and
    connect_{in|out} methods and setsockopt_{in|out} allow users to specify
    connections for the sockets.

    Parameters
    ----------
    device_type : int
        The 0MQ Device type
    {in|out}_type : int
        zmq socket types, to be passed later to context.socket(). e.g.
        zmq.PUB, zmq.SUB, zmq.REQ. If out_type is < 0, then in_socket is used
        for both in_socket and out_socket.

    Methods
    -------
    bind_{in_out}(iface)
        passthrough for ``{in|out}_socket.bind(iface)``, to be called in the thread
    connect_{in_out}(iface)
        passthrough for ``{in|out}_socket.connect(iface)``, to be called in the
        thread
    setsockopt_{in_out}(opt,value)
        passthrough for ``{in|out}_socket.setsockopt(opt, value)``, to be called in
        the thread

    Attributes
    ----------
    daemon : bool
        sets whether the thread should be run as a daemon
        Default is true, because if it is false, the thread will not
        exit unless it is killed
    context_factory : callable (class attribute)
        Function for creating the Context. This will be Context.instance
        in ThreadDevices, and Context in ProcessDevices.  The only reason
        it is not instance() in ProcessDevices is that there may be a stale
        Context instance already initialized, and the forked environment
        should *never* try to use it.
    """
    context_factory: Callable[[], zmq.Context] = Context.instance
    'Callable that returns a context. Typically either Context.instance or Context,\n    depending on whether the device should share the global instance or not.\n    '
    daemon: bool
    device_type: int
    in_type: int
    out_type: int
    _in_binds: List[str]
    _in_connects: List[str]
    _in_sockopts: List[Tuple[int, Any]]
    _out_binds: List[str]
    _out_connects: List[str]
    _out_sockopts: List[Tuple[int, Any]]
    _random_addrs: List[str]
    _sockets: List[zmq.Socket]

    def __init__(self, device_type: int=QUEUE, in_type: Optional[int]=None, out_type: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        self.device_type = device_type
        if in_type is None:
            raise TypeError('in_type must be specified')
        if out_type is None:
            raise TypeError('out_type must be specified')
        self.in_type = in_type
        self.out_type = out_type
        self._in_binds = []
        self._in_connects = []
        self._in_sockopts = []
        self._out_binds = []
        self._out_connects = []
        self._out_sockopts = []
        self._random_addrs = []
        self.daemon = True
        self.done = False
        self._sockets = []

    def bind_in(self, addr: str) -> None:
        if False:
            print('Hello World!')
        'Enqueue ZMQ address for binding on in_socket.\n\n        See zmq.Socket.bind for details.\n        '
        self._in_binds.append(addr)

    def bind_in_to_random_port(self, addr: str, *args, **kwargs) -> int:
        if False:
            print('Hello World!')
        'Enqueue a random port on the given interface for binding on\n        in_socket.\n\n        See zmq.Socket.bind_to_random_port for details.\n\n        .. versionadded:: 18.0\n        '
        port = self._reserve_random_port(addr, *args, **kwargs)
        self.bind_in('%s:%i' % (addr, port))
        return port

    def connect_in(self, addr: str) -> None:
        if False:
            i = 10
            return i + 15
        'Enqueue ZMQ address for connecting on in_socket.\n\n        See zmq.Socket.connect for details.\n        '
        self._in_connects.append(addr)

    def setsockopt_in(self, opt: int, value: Any) -> None:
        if False:
            while True:
                i = 10
        'Enqueue setsockopt(opt, value) for in_socket\n\n        See zmq.Socket.setsockopt for details.\n        '
        self._in_sockopts.append((opt, value))

    def bind_out(self, addr: str) -> None:
        if False:
            i = 10
            return i + 15
        'Enqueue ZMQ address for binding on out_socket.\n\n        See zmq.Socket.bind for details.\n        '
        self._out_binds.append(addr)

    def bind_out_to_random_port(self, addr: str, *args, **kwargs) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Enqueue a random port on the given interface for binding on\n        out_socket.\n\n        See zmq.Socket.bind_to_random_port for details.\n\n        .. versionadded:: 18.0\n        '
        port = self._reserve_random_port(addr, *args, **kwargs)
        self.bind_out('%s:%i' % (addr, port))
        return port

    def connect_out(self, addr: str):
        if False:
            return 10
        'Enqueue ZMQ address for connecting on out_socket.\n\n        See zmq.Socket.connect for details.\n        '
        self._out_connects.append(addr)

    def setsockopt_out(self, opt: int, value: Any):
        if False:
            i = 10
            return i + 15
        'Enqueue setsockopt(opt, value) for out_socket\n\n        See zmq.Socket.setsockopt for details.\n        '
        self._out_sockopts.append((opt, value))

    def _reserve_random_port(self, addr: str, *args, **kwargs) -> int:
        if False:
            i = 10
            return i + 15
        with Context() as ctx:
            with ctx.socket(PUSH) as binder:
                for i in range(5):
                    port = binder.bind_to_random_port(addr, *args, **kwargs)
                    new_addr = '%s:%i' % (addr, port)
                    if new_addr in self._random_addrs:
                        continue
                    else:
                        break
                else:
                    raise ZMQBindError('Could not reserve random port.')
                self._random_addrs.append(new_addr)
        return port

    def _setup_sockets(self) -> Tuple[zmq.Socket, zmq.Socket]:
        if False:
            while True:
                i = 10
        ctx: zmq.Context[zmq.Socket] = self.context_factory()
        self._context = ctx
        ins = ctx.socket(self.in_type)
        self._sockets.append(ins)
        if self.out_type < 0:
            outs = ins
        else:
            outs = ctx.socket(self.out_type)
            self._sockets.append(outs)
        for (opt, value) in self._in_sockopts:
            ins.setsockopt(opt, value)
        for (opt, value) in self._out_sockopts:
            outs.setsockopt(opt, value)
        for iface in self._in_binds:
            ins.bind(iface)
        for iface in self._out_binds:
            outs.bind(iface)
        for iface in self._in_connects:
            ins.connect(iface)
        for iface in self._out_connects:
            outs.connect(iface)
        return (ins, outs)

    def run_device(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'The runner method.\n\n        Do not call me directly, instead call ``self.start()``, just like a Thread.\n        '
        (ins, outs) = self._setup_sockets()
        device(self.device_type, ins, outs)

    def _close_sockets(self):
        if False:
            for i in range(10):
                print('nop')
        'Cleanup sockets we created'
        for s in self._sockets:
            if s and (not s.closed):
                s.close()

    def run(self) -> None:
        if False:
            print('Hello World!')
        'wrap run_device in try/catch ETERM'
        try:
            self.run_device()
        except ZMQError as e:
            if e.errno in {ETERM, ENOTSOCK}:
                pass
            else:
                raise
        finally:
            self.done = True
            self._close_sockets()

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start the device. Override me in subclass for other launchers.'
        return self.run()

    def join(self, timeout: Optional[float]=None) -> None:
        if False:
            i = 10
            return i + 15
        'wait for me to finish, like Thread.join.\n\n        Reimplemented appropriately by subclasses.'
        tic = time.monotonic()
        toc = tic
        while not self.done and (not (timeout is not None and toc - tic > timeout)):
            time.sleep(0.001)
            toc = time.monotonic()

class BackgroundDevice(Device):
    """Base class for launching Devices in background processes and threads."""
    launcher: Any = None
    _launch_class: Any = None

    def start(self) -> None:
        if False:
            while True:
                i = 10
        self.launcher = self._launch_class(target=self.run)
        self.launcher.daemon = self.daemon
        return self.launcher.start()

    def join(self, timeout: Optional[float]=None) -> None:
        if False:
            return 10
        return self.launcher.join(timeout=timeout)

class ThreadDevice(BackgroundDevice):
    """A Device that will be run in a background Thread.

    See Device for details.
    """
    _launch_class = Thread

class ProcessDevice(BackgroundDevice):
    """A Device that will be run in a background Process.

    See Device for details.
    """
    _launch_class = Process
    context_factory = Context
    'Callable that returns a context. Typically either Context.instance or Context,\n    depending on whether the device should share the global instance or not.\n    '
__all__ = ['Device', 'ThreadDevice', 'ProcessDevice']