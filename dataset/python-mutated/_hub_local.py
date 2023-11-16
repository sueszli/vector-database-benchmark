"""
Maintains the thread local hub.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _thread
__all__ = ['get_hub', 'get_hub_noargs', 'get_hub_if_exists']
assert 'gevent' not in str(_thread._local)

class _Threadlocal(_thread._local):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(_Threadlocal, self).__init__()
        self.Hub = None
        self.loop = None
        self.hub = None
_threadlocal = _Threadlocal()
Hub = None

def get_hub_class():
    if False:
        while True:
            i = 10
    "Return the type of hub to use for the current thread.\n\n    If there's no type of hub for the current thread yet, 'gevent.hub.Hub' is used.\n    "
    hubtype = _threadlocal.Hub
    if hubtype is None:
        hubtype = _threadlocal.Hub = Hub
    return hubtype

def set_default_hub_class(hubtype):
    if False:
        return 10
    global Hub
    Hub = hubtype

def get_hub():
    if False:
        i = 10
        return i + 15
    '\n    Return the hub for the current thread.\n\n    If a hub does not exist in the current thread, a new one is\n    created of the type returned by :func:`get_hub_class`.\n\n    .. deprecated:: 1.3b1\n       The ``*args`` and ``**kwargs`` arguments are deprecated. They were\n       only used when the hub was created, and so were non-deterministic---to be\n       sure they were used, *all* callers had to pass them, or they were order-dependent.\n       Use ``set_hub`` instead.\n\n    .. versionchanged:: 1.5a3\n       The *args* and *kwargs* arguments are now completely ignored.\n\n    .. versionchanged:: 23.7.0\n       The long-deprecated ``args`` and ``kwargs`` parameters are no\n       longer accepted.\n    '
    try:
        hub = _threadlocal.hub
    except AttributeError:
        hub = None
    if hub is None:
        hubtype = get_hub_class()
        hub = _threadlocal.hub = hubtype()
    return hub

def get_hub_noargs():
    if False:
        return 10
    try:
        hub = _threadlocal.hub
    except AttributeError:
        hub = None
    if hub is None:
        hubtype = get_hub_class()
        hub = _threadlocal.hub = hubtype()
    return hub

def get_hub_if_exists():
    if False:
        i = 10
        return i + 15
    '\n    Return the hub for the current thread.\n\n    Return ``None`` if no hub has been created yet.\n    '
    try:
        return _threadlocal.hub
    except AttributeError:
        return None

def set_hub(hub):
    if False:
        print('Hello World!')
    _threadlocal.hub = hub

def get_loop():
    if False:
        i = 10
        return i + 15
    return _threadlocal.loop

def set_loop(loop):
    if False:
        for i in range(10):
            print('nop')
    _threadlocal.loop = loop
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent.__hub_local')