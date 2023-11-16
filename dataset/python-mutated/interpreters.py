"""Subinterpreters High Level Module."""
import time
import _xxsubinterpreters as _interpreters
from _xxsubinterpreters import ChannelError, ChannelNotFoundError, ChannelEmptyError, is_shareable
__all__ = ['Interpreter', 'get_current', 'get_main', 'create', 'list_all', 'SendChannel', 'RecvChannel', 'create_channel', 'list_all_channels', 'is_shareable', 'ChannelError', 'ChannelNotFoundError', 'ChannelEmptyError']

def create(*, isolated=True):
    if False:
        while True:
            i = 10
    'Return a new (idle) Python interpreter.'
    id = _interpreters.create(isolated=isolated)
    return Interpreter(id, isolated=isolated)

def list_all():
    if False:
        print('Hello World!')
    'Return all existing interpreters.'
    return [Interpreter(id) for id in _interpreters.list_all()]

def get_current():
    if False:
        for i in range(10):
            print('nop')
    'Return the currently running interpreter.'
    id = _interpreters.get_current()
    return Interpreter(id)

def get_main():
    if False:
        i = 10
        return i + 15
    'Return the main interpreter.'
    id = _interpreters.get_main()
    return Interpreter(id)

class Interpreter:
    """A single Python interpreter."""

    def __init__(self, id, *, isolated=None):
        if False:
            while True:
                i = 10
        if not isinstance(id, (int, _interpreters.InterpreterID)):
            raise TypeError(f'id must be an int, got {id!r}')
        self._id = id
        self._isolated = isolated

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        data = dict(id=int(self._id), isolated=self._isolated)
        kwargs = (f'{k}={v!r}' for (k, v) in data.items())
        return f"{type(self).__name__}({', '.join(kwargs)})"

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self._id)

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, Interpreter):
            return NotImplemented
        else:
            return other._id == self._id

    @property
    def id(self):
        if False:
            print('Hello World!')
        return self._id

    @property
    def isolated(self):
        if False:
            return 10
        if self._isolated is None:
            self._isolated = _interpreters.is_isolated(self._id)
        return self._isolated

    def is_running(self):
        if False:
            for i in range(10):
                print('nop')
        'Return whether or not the identified interpreter is running.'
        return _interpreters.is_running(self._id)

    def close(self):
        if False:
            i = 10
            return i + 15
        'Finalize and destroy the interpreter.\n\n        Attempting to destroy the current interpreter results\n        in a RuntimeError.\n        '
        return _interpreters.destroy(self._id)

    def run(self, src_str, /, *, channels=None):
        if False:
            i = 10
            return i + 15
        'Run the given source code in the interpreter.\n\n        This blocks the current Python thread until done.\n        '
        _interpreters.run_string(self._id, src_str, channels)

def create_channel():
    if False:
        i = 10
        return i + 15
    'Return (recv, send) for a new cross-interpreter channel.\n\n    The channel may be used to pass data safely between interpreters.\n    '
    cid = _interpreters.channel_create()
    (recv, send) = (RecvChannel(cid), SendChannel(cid))
    return (recv, send)

def list_all_channels():
    if False:
        return 10
    'Return a list of (recv, send) for all open channels.'
    return [(RecvChannel(cid), SendChannel(cid)) for cid in _interpreters.channel_list_all()]

class _ChannelEnd:
    """The base class for RecvChannel and SendChannel."""

    def __init__(self, id):
        if False:
            return 10
        if not isinstance(id, (int, _interpreters.ChannelID)):
            raise TypeError(f'id must be an int, got {id!r}')
        self._id = id

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{type(self).__name__}(id={int(self._id)})'

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self._id)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(self, RecvChannel):
            if not isinstance(other, RecvChannel):
                return NotImplemented
        elif not isinstance(other, SendChannel):
            return NotImplemented
        return other._id == self._id

    @property
    def id(self):
        if False:
            while True:
                i = 10
        return self._id
_NOT_SET = object()

class RecvChannel(_ChannelEnd):
    """The receiving end of a cross-interpreter channel."""

    def recv(self, *, _sentinel=object(), _delay=10 / 1000):
        if False:
            for i in range(10):
                print('nop')
        'Return the next object from the channel.\n\n        This blocks until an object has been sent, if none have been\n        sent already.\n        '
        obj = _interpreters.channel_recv(self._id, _sentinel)
        while obj is _sentinel:
            time.sleep(_delay)
            obj = _interpreters.channel_recv(self._id, _sentinel)
        return obj

    def recv_nowait(self, default=_NOT_SET):
        if False:
            i = 10
            return i + 15
        'Return the next object from the channel.\n\n        If none have been sent then return the default if one\n        is provided or fail with ChannelEmptyError.  Otherwise this\n        is the same as recv().\n        '
        if default is _NOT_SET:
            return _interpreters.channel_recv(self._id)
        else:
            return _interpreters.channel_recv(self._id, default)

class SendChannel(_ChannelEnd):
    """The sending end of a cross-interpreter channel."""

    def send(self, obj):
        if False:
            return 10
        "Send the object (i.e. its data) to the channel's receiving end.\n\n        This blocks until the object is received.\n        "
        _interpreters.channel_send(self._id, obj)
        time.sleep(2)

    def send_nowait(self, obj):
        if False:
            print('Hello World!')
        "Send the object to the channel's receiving end.\n\n        If the object is immediately received then return True\n        (else False).  Otherwise this is the same as send().\n        "
        return _interpreters.channel_send(self._id, obj)