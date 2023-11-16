"""
Implements the emitter decorator, class and desciptor.
"""
import weakref
from ._action import BaseDescriptor

def emitter(func):
    if False:
        return 10
    " Decorator to turn a method of a Component into an\n    :class:`Emitter <flexx.event.Emitter>`.\n\n    An emitter makes it easy to emit specific events, and is also a\n    placeholder for documenting an event.\n\n    .. code-block:: python\n\n        class MyObject(event.Component):\n\n           @emitter\n           def spam(self, v):\n                return dict(value=v)\n\n        m = MyObject()\n        m.spam(42)  # emit the spam event\n\n    The method being decorated can have any number of arguments, and\n    should return a dictionary that represents the event to generate.\n    The method's docstring is used as the emitter's docstring.\n    "
    if not callable(func):
        raise TypeError('The event.emitter() decorator needs a function.')
    if getattr(func, '__self__', None) is not None:
        raise TypeError('Invalid use of emitter decorator.')
    return EmitterDescriptor(func, func.__name__, func.__doc__)

class EmitterDescriptor(BaseDescriptor):
    """ Placeholder for documentation and easy emitting of the event.
    """

    def __init__(self, func, name, doc):
        if False:
            for i in range(10):
                print('nop')
        self._func = func
        self._name = name
        self.__doc__ = self._format_doc('emitter', name, doc, func)

    def __get__(self, instance, owner):
        if False:
            while True:
                i = 10
        if instance is None:
            return self
        private_name = '_' + self._name + '_emitter'
        try:
            emitter = getattr(instance, private_name)
        except AttributeError:
            emitter = Emitter(instance, self._func, self._name, self.__doc__)
            setattr(instance, private_name, emitter)
        emitter._use_once(self._func)
        return emitter

class Emitter:
    """ Emitter objects are wrappers around Component methods. They take
    care of emitting an event when called and function as a placeholder
    for documenting an event. This class should not be instantiated
    directly; use ``event.emitter()`` instead.
    """

    def __init__(self, ob, func, name, doc):
        if False:
            return 10
        assert callable(func)
        self._ob1 = weakref.ref(ob)
        self._func = func
        self._func_once = func
        self._name = name
        self.__doc__ = doc

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        cname = self.__class__.__name__
        return '<%s %r at 0x%x>' % (cname, self._name, id(self))

    def _use_once(self, func):
        if False:
            return 10
        ' To support super().\n        '
        self._func_once = func

    def __call__(self, *args):
        if False:
            print('Hello World!')
        ' Emit the event.\n        '
        func = self._func_once
        self._func_once = self._func
        ob = self._ob1()
        if ob is not None:
            ev = func(ob, *args)
            if ev is not None:
                ob.emit(self._name, ev)