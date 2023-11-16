"""
Implements the action decorator, class and desciptor.
"""
import weakref
import inspect
from ._loop import loop
from . import logger

def action(func):
    if False:
        while True:
            i = 10
    ' Decorator to turn a method of a Component into an\n    :class:`Action <flexx.event.Action>`.\n\n    Actions change the state of the application by\n    :func:`mutating <flexx.event.Component._mutate>`\n    :class:`properties <flexx.event.Property>`.\n    In fact, properties can only be changed via actions.\n\n    Actions are asynchronous and thread-safe. Invoking an action will not\n    apply the changes directly; the action is queued and handled at a later\n    time. The one exception is that when an action is invoked from anoher\n    action, it is handled directly.\n\n    Although setting properties directly might seem nice, their use would mean\n    that the state of the application can change while the app is *reacting*\n    to changes in the state. This might be managable for small applications,\n    but as an app grows this easily results in inconsistencies and bugs.\n    Separating actions (which modify state) and reactions (that react to it)\n    makes apps easier to understand and debug. This is the core idea behind\n    frameworks such as Elm, React and Veux. And Flexx adopts it as well.\n\n    Example usage:\n\n    .. code-block:: py\n\n        class MyComponent(event.Component):\n\n            count = event.IntProp(0)\n\n            @action\n            def increase_counter(self):\n                self._mutate_count(self.count + 1)  # call mutator function\n\n    '
    if not callable(func):
        raise TypeError('The event.action() decorator needs a function.')
    if getattr(func, '__self__', None) is not None:
        raise TypeError('Invalid use of action decorator.')
    return ActionDescriptor(func, func.__name__, func.__doc__ or func.__name__)

class BaseDescriptor:
    """ Base descriptor class for some commonalities.
    """

    def __repr__(self):
        if False:
            while True:
                i = 10
        t = '<%s %r (this should be a class attribute) at 0x%x>'
        return t % (self.__class__.__name__, self._name, id(self))

    def __set__(self, obj, value):
        if False:
            for i in range(10):
                print('nop')
        cname = self.__class__.__name__
        cname = cname[:-10] if cname.endswith('Descriptor') else cname
        raise AttributeError('Cannot overwrite %s %r.' % (cname, self._name))

    def __delete__(self, obj):
        if False:
            for i in range(10):
                print('nop')
        cname = self.__class__.__name__
        cname = cname[:-10] if cname.endswith('Descriptor') else cname
        raise AttributeError('Cannot delete %s %r.' % (cname, self._name))

    @staticmethod
    def _format_doc(kind, name, doc, func=None):
        if False:
            i = 10
            return i + 15
        (prefix, betweenfix) = ('', ' ')
        doc = (doc or '').strip()
        if doc.count('\n') and doc.split('\n')[0].strip().count(':'):
            line2 = doc.split('\n')[1]
            betweenfix = '\n' + ' ' * (len(line2) - len(line2.lstrip()))
        if doc:
            if func:
                sig = str(inspect.signature(func))
                sig = '(' + sig[5:].lstrip(', ') if sig.startswith('(self') else sig
                prefix = '{}{}\n'.format(name, sig)
            return '{}*{}* –{}{}\n'.format(prefix, kind, betweenfix, doc or name)

class ActionDescriptor(BaseDescriptor):
    """ Class descriptor for actions.
    """

    def __init__(self, func, name, doc):
        if False:
            for i in range(10):
                print('nop')
        self._func = func
        self._name = name
        self.__doc__ = self._format_doc('action', name, doc, func)

    def __get__(self, instance, owner):
        if False:
            i = 10
            return i + 15
        if instance is None:
            return self
        private_name = '_' + self._name + '_action'
        try:
            action = getattr(instance, private_name)
        except AttributeError:
            action = Action(instance, self._func, self._name, self.__doc__)
            setattr(instance, private_name, action)
        action._use_once(self._func)
        return action

class Action:
    """ Action objects are wrappers around Component methods. They take
    care of queueing action invokations rather than calling the function
    directly, unless the action is called from another action (in this
    case it would a direct call). This class should not be instantiated
    directly; use ``event.action()`` instead.
    """

    def __init__(self, ob, func, name, doc):
        if False:
            for i in range(10):
                print('nop')
        assert callable(func)
        self._ob1 = weakref.ref(ob)
        self._func = func
        self._func_once = func
        self._name = name
        self.__doc__ = doc
        self.is_autogenerated = func.__name__ == 'flx_setter'

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        cname = self.__class__.__name__
        return '<%s %r at 0x%x>' % (cname, self._name, id(self))

    def _use_once(self, func):
        if False:
            for i in range(10):
                print('nop')
        ' To support super().\n        '
        self._func_once = func

    def __call__(self, *args):
        if False:
            i = 10
            return i + 15
        ' Invoke the action.\n        '
        ob = self._ob1()
        if loop.can_mutate(ob):
            func = self._func_once
            self._func_once = self._func
            if ob is not None:
                res = func(ob, *args)
                if res is not None:
                    logger.warning('Action (%s) should not return a value' % self._name)
        else:
            loop.add_action_invokation(self, args)
        return ob