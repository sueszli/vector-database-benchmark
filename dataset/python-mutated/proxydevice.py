"""Proxy classes and functions."""
import zmq
from zmq.devices.basedevice import Device, ProcessDevice, ThreadDevice

class ProxyBase:
    """Base class for overriding methods."""

    def __init__(self, in_type, out_type, mon_type=zmq.PUB):
        if False:
            return 10
        Device.__init__(self, in_type=in_type, out_type=out_type)
        self.mon_type = mon_type
        self._mon_binds = []
        self._mon_connects = []
        self._mon_sockopts = []

    def bind_mon(self, addr):
        if False:
            i = 10
            return i + 15
        'Enqueue ZMQ address for binding on mon_socket.\n\n        See zmq.Socket.bind for details.\n        '
        self._mon_binds.append(addr)

    def bind_mon_to_random_port(self, addr, *args, **kwargs):
        if False:
            print('Hello World!')
        'Enqueue a random port on the given interface for binding on\n        mon_socket.\n\n        See zmq.Socket.bind_to_random_port for details.\n\n        .. versionadded:: 18.0\n        '
        port = self._reserve_random_port(addr, *args, **kwargs)
        self.bind_mon('%s:%i' % (addr, port))
        return port

    def connect_mon(self, addr):
        if False:
            print('Hello World!')
        'Enqueue ZMQ address for connecting on mon_socket.\n\n        See zmq.Socket.connect for details.\n        '
        self._mon_connects.append(addr)

    def setsockopt_mon(self, opt, value):
        if False:
            i = 10
            return i + 15
        'Enqueue setsockopt(opt, value) for mon_socket\n\n        See zmq.Socket.setsockopt for details.\n        '
        self._mon_sockopts.append((opt, value))

    def _setup_sockets(self):
        if False:
            while True:
                i = 10
        (ins, outs) = Device._setup_sockets(self)
        ctx = self._context
        mons = ctx.socket(self.mon_type)
        self._sockets.append(mons)
        for (opt, value) in self._mon_sockopts:
            mons.setsockopt(opt, value)
        for iface in self._mon_binds:
            mons.bind(iface)
        for iface in self._mon_connects:
            mons.connect(iface)
        return (ins, outs, mons)

    def run_device(self):
        if False:
            i = 10
            return i + 15
        (ins, outs, mons) = self._setup_sockets()
        zmq.proxy(ins, outs, mons)

class Proxy(ProxyBase, Device):
    """Threadsafe Proxy object.

    See zmq.devices.Device for most of the spec. This subclass adds a
    <method>_mon version of each <method>_{in|out} method, for configuring the
    monitor socket.

    A Proxy is a 3-socket ZMQ Device that functions just like a
    QUEUE, except each message is also sent out on the monitor socket.

    A PUB socket is the most logical choice for the mon_socket, but it is not required.
    """

class ThreadProxy(ProxyBase, ThreadDevice):
    """Proxy in a Thread. See Proxy for more."""

class ProcessProxy(ProxyBase, ProcessDevice):
    """Proxy in a Process. See Proxy for more."""
__all__ = ['Proxy', 'ThreadProxy', 'ProcessProxy']