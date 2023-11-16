"""
A collection of primitives used by the hub, and suitable for
compilation with Cython because of their frequency of use.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from weakref import ref as wref
from gc import get_objects
from greenlet import greenlet
from gevent.exceptions import BlockingSwitchOutError
locals()['getcurrent'] = __import__('greenlet').getcurrent
locals()['greenlet_init'] = lambda : None
locals()['_greenlet_switch'] = greenlet.switch
__all__ = ['TrackedRawGreenlet', 'SwitchOutGreenletWithLoop']

class TrackedRawGreenlet(greenlet):

    def __init__(self, function, parent):
        if False:
            print('Hello World!')
        greenlet.__init__(self, function, parent)
        current = getcurrent()
        self.spawning_greenlet = wref(current)
        try:
            self.spawn_tree_locals = current.spawn_tree_locals
        except AttributeError:
            self.spawn_tree_locals = {}
            if current.parent:
                current.spawn_tree_locals = self.spawn_tree_locals

class SwitchOutGreenletWithLoop(TrackedRawGreenlet):

    def switch(self):
        if False:
            while True:
                i = 10
        switch_out = getattr(getcurrent(), 'switch_out', None)
        if switch_out is not None:
            switch_out()
        return _greenlet_switch(self)

    def switch_out(self):
        if False:
            while True:
                i = 10
        raise BlockingSwitchOutError('Impossible to call blocking function in the event loop callback')

def get_reachable_greenlets():
    if False:
        for i in range(10):
            print('nop')
    return [x for x in get_objects() if isinstance(x, greenlet) and (not getattr(x, 'greenlet_tree_is_ignored', False))]
_memoryview = memoryview
try:
    if isinstance(__builtins__, dict):
        _buffer = __builtins__['buffer']
    else:
        _buffer = __builtins__.buffer
except (AttributeError, KeyError):
    _buffer = memoryview

def get_memory(data):
    if False:
        print('Hello World!')
    try:
        mv = _memoryview(data) if not isinstance(data, _memoryview) else data
        if mv.shape:
            return mv
        return mv.tobytes()
    except TypeError:
        if _buffer is _memoryview:
            raise
        return _buffer(data)

def _init():
    if False:
        return 10
    greenlet_init()
_init()
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent.__greenlet_primitives')