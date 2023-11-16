"""
Implements the reaction decorator, class and desciptor.
"""
import weakref
import inspect
from ._loop import this_is_js
from ._action import BaseDescriptor
from ._dict import Dict
from . import logger
window = None
console = logger

def looks_like_method(func):
    if False:
        print('Hello World!')
    if hasattr(func, '__func__'):
        return False
    try:
        return list(inspect.signature(func).parameters)[0] in ('self', 'this')
    except (TypeError, IndexError, ValueError):
        return False

def reaction(*connection_strings, mode='normal'):
    if False:
        return 10
    ' Decorator to turn a method of a Component into a\n    :class:`Reaction <flexx.event.Reaction>`.\n\n    A reaction can be connected to multiple event types. Each connection\n    string represents an event type to connect to.\n\n    Also see the\n    :func:`Component.reaction() <flexx.event.Component.reaction>` method.\n\n    .. code-block:: py\n\n        class MyObject(event.Component):\n\n            @event.reaction(\'first_name\', \'last_name\')\n            def greet(self, *events):\n                print(\'hello %s %s\' % (self.first_name, self.last_name))\n\n    A reaction can operate in a few different modes. By not specifying any\n    connection strings, the mode is "auto": the reaction will automatically\n    trigger when any of the properties used in the function changes.\n    See :func:`get_mode() <flexx.event.Reaction.get_mode>` for details.\n    \n    Connection string follow the following syntax rules:\n    \n    * Connection strings consist of parts separated by dots, thus forming a path.\n      If an element on the path is a property, the connection will automatically\n      reset when that property changes (a.k.a. dynamism, more on this below).\n    * Each part can end with one star (\'*\'), indicating that the part is a list\n      and that a connection should be made for each item in the list.\n    * With two stars, the connection is made *recursively*, e.g. "children**"\n      connects to "children" and the children\'s children, etc.\n    * Stripped of \'*\', each part must be a valid identifier (ASCII).\n    * The total string optionally has a label suffix separated by a colon. The\n      label itself may consist of any characters.\n    * The string can have a "!" at the very start to suppress warnings for\n      connections to event types that Flexx is not aware of at initialization\n      time (i.e. not corresponding to a property or emitter).\n    \n    An extreme example could be ``"!foo.children**.text:mylabel"``, which connects\n    to the "text" event of the children (and their children, and their children\'s\n    children etc.) of the ``foo`` attribute. The "!" is common in cases like\n    this to suppress warnings if not all children have a ``text`` event/property.\n    \n    '
    if not connection_strings:
        raise TypeError('reaction() needs one or more arguments.')
    mode = mode or 'normal'
    if not isinstance(mode, str):
        raise TypeError('Reaction mode must be a string.')
    mode = mode.lower()
    if mode not in ('normal', 'greedy', 'auto'):
        raise TypeError('Reaction mode must "normal", "greedy" or "auto".')
    func = None
    if len(connection_strings) == 1 and callable(connection_strings[0]):
        func = connection_strings[0]
        connection_strings = []
    for s in connection_strings:
        if not (isinstance(s, str) and len(s) > 0):
            raise TypeError('Connection string must be nonempty strings.')

    def _connect(func):
        if False:
            i = 10
            return i + 15
        if not callable(func):
            raise TypeError('reaction() decorator requires a callable.')
        if not looks_like_method(func):
            raise TypeError('reaction() decorator requires a method (first arg must be self).')
        return ReactionDescriptor(func, mode, connection_strings)
    if func is not None:
        return _connect(func)
    else:
        return _connect

class ReactionDescriptor(BaseDescriptor):
    """ Class descriptor for reactions.
    """

    def __init__(self, func, mode, connection_strings, ob=None):
        if False:
            i = 10
            return i + 15
        self._name = func.__name__
        self._func = func
        self._mode = mode
        if len(connection_strings) == 0:
            self._mode = 'auto'
        self._connection_strings = connection_strings
        self._ob = None if ob is None else weakref.ref(ob)
        self.__doc__ = self._format_doc('reaction', self._name, func.__doc__)

    def __get__(self, instance, owner):
        if False:
            return 10
        if instance is None:
            return self
        private_name = '_' + self._name + '_reaction'
        try:
            reaction = getattr(instance, private_name)
        except AttributeError:
            reaction = Reaction(instance if self._ob is None else self._ob(), (self._func, instance), self._mode, self._connection_strings)
            setattr(instance, private_name, reaction)
        reaction._use_once(self._func)
        return reaction

    @property
    def local_connection_strings(self):
        if False:
            return 10
        ' List of connection strings that are local to the object.\n        '
        return [s for s in self._connection_strings if '.' not in s]

class Reaction:
    """ Reaction objects are wrappers around Component methods. They connect
    to one or more events. This class should not be instantiated directly;
    use ``event.reaction()`` or ``Component.reaction()`` instead.
    """
    _count = 0

    def __init__(self, ob, func, mode, connection_strings):
        if False:
            return 10
        Reaction._count += 1
        self._id = 'r%i' % Reaction._count
        self._ob1 = weakref.ref(ob)
        self._ob2 = None
        if isinstance(func, tuple):
            self._ob2 = weakref.ref(func[1])
            func = func[0]
        if getattr(func, '__self__', None) is not None:
            if getattr(func, '__func__', None) is not None:
                self._ob2 = weakref.ref(func.__self__)
                func = func.__func__
        assert callable(func)
        assert mode in ('normal', 'greedy', 'auto')
        self._func = func
        self._func_once = func
        self._mode = mode
        self._name = func.__name__
        self.__doc__ = BaseDescriptor._format_doc('reaction', self._name, func.__doc__)
        self._init(connection_strings)

    def _init(self, connection_strings):
        if False:
            i = 10
            return i + 15
        ' Init of this reaction that is compatible with PScript.\n        '
        ichars = '0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self._connections = []
        self._implicit_connections = []
        for ic in range(len(connection_strings)):
            fullname = connection_strings[ic]
            force = fullname.startswith('!')
            (s, _, label) = fullname.lstrip('!').partition(':')
            s0 = s
            if '.*.' in s + '.':
                s = s.replace('.*', '*')
                console.warn('Connection string syntax "foo.*.bar" is deprecated, use "%s" instead of "%s":.' % (s, s0))
            if '!' in s:
                s = s.replace('!', '')
                force = True
                console.warn('Exclamation marks in connection strings must come at the very start, use "!%s" instead of "%s".' % (s, s0))
            parts = s.split('.')
            for ipart in range(len(parts)):
                part = parts[ipart].rstrip('*')
                is_identifier = len(part) > 0
                for i in range(len(part)):
                    is_identifier = is_identifier and part[i] in ichars
                if is_identifier is False:
                    raise ValueError('Connection string %r contains non-identifier part %r' % (s, part))
            d = Dict()
            self._connections.append(d)
            d.fullname = fullname
            d.parts = parts
            d.type = parts[-1].rstrip('*') + ':' + (label or self._name)
            d.force = force
            d.objects = []
        for ic in range(len(self._connections)):
            self.reconnect(ic)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        c = '+'.join([str(len(c.objects)) for c in self._connections])
        cname = self.__class__.__name__
        t = '<%s %r (%s) with %s connections at 0x%x>'
        return t % (cname, self._name, self._mode, c, id(self))

    def get_mode(self):
        if False:
            i = 10
            return i + 15
        " Get the mode for this reaction:\n\n        * 'normal': events are handled in the order that they were emitted.\n          Consequently, there can be multiple calls per event loop iteration\n          if other reactions were triggered as well.\n        * 'greedy': this reaction receives all its events (since the last event\n          loop iteration) in a single call (even if this breaks the order of\n          events with respect to other reactions). Use this when multiple related\n          events must be handled simultaneously (e.g. when syncing properties).\n        * 'auto': this reaction tracks what properties it uses, and is\n          automatically triggered when any of these properties changes. Like\n          'greedy' there is at most one call per event loop iteration.\n          Reactions with zero connection strings always have mode 'auto'.\n\n        The 'normal' mode generally offers the most consistent behaviour.\n        The 'greedy' mode allows the event system to make some optimizations.\n        Combined with the fact that there is at most one call per event loop\n        iteration, this can provide higher performance where it matters.\n        Reactions with mode 'auto' can be a convenient way to connect things\n        up. Although it allows the event system to make the same optimizations\n        as 'greedy', it also needs to reconnect the reaction after each time\n        it is called, which can degregade performance especially if many\n        properties are accessed by the reaction.\n        "
        return self._mode

    def get_name(self):
        if False:
            i = 10
            return i + 15
        ' Get the name of this reaction, usually corresponding to the name\n        of the function that this reaction wraps.\n        '
        return self._name

    def get_connection_info(self):
        if False:
            print('Hello World!')
        ' Get a list of tuples (name, connection_names), where\n        connection_names is a list of type names (including label) for\n        the made connections.\n        '
        return [(c.fullname, [u[1] for u in c.objects]) for c in self._connections]

    def _use_once(self, func):
        if False:
            i = 10
            return i + 15
        self._func_once = func

    def __call__(self, *events):
        if False:
            for i in range(10):
                print('nop')
        ' Call the reaction function.\n        '
        func = self._func_once
        self._func_once = self._func
        if self._ob2 is not None:
            if self._ob2() is not None:
                res = func(self._ob2(), *events)
            else:
                self.dispose()
                return
        else:
            res = func(*events)
        return res

    def dispose(self):
        if False:
            for i in range(10):
                print('nop')
        ' Disconnect all connections so that there are no more references\n        to components.\n        '
        if len(self._connections) == 0 and len(self._implicit_connections) == 0:
            return
        if not this_is_js():
            self._ob1 = lambda : None
            logger.debug('Disposing reaction %r ' % self)
        while len(self._implicit_connections):
            (ob, type) = self._implicit_connections.pop(0)
            ob.disconnect(type, self)
        for ic in range(len(self._connections)):
            connection = self._connections[ic]
            while len(connection.objects) > 0:
                (ob, type) = connection.objects.pop(0)
                ob.disconnect(type, self)
        self._connections = []

    def _update_implicit_connections(self, connections):
        if False:
            while True:
                i = 10
        ' Update the list of implicit (i.e. automatic) connections.\n        Used by the loop.\n        '
        old_conns = self._implicit_connections
        new_conns = connections
        self._implicit_connections = new_conns
        self._connect_and_disconnect(old_conns, new_conns)

    def _clear_component_refs(self, ob):
        if False:
            print('Hello World!')
        " Clear all references to the given Component instance. This is\n        called from a Component' dispose() method. This reaction remains\n        working, but wont receive events from that object anymore.\n        "
        for i in range(len(self._implicit_connections) - 1, -1, -1):
            if self._implicit_connections[i][0] is ob:
                self._implicit_connections.pop(i)
        for ic in range(len(self._connections)):
            connection = self._connections[ic]
            for i in range(len(connection.objects) - 1, -1, -1):
                if connection.objects[i][0] is ob:
                    connection.objects.pop(i)

    def reconnect(self, index):
        if False:
            print('Hello World!')
        " (re)connect the index'th connection.\n        "
        connection = self._connections[index]
        old_objects = connection.objects
        connection.objects = []
        ob = self._ob1()
        if ob is not None:
            self._seek_event_object(index, connection.parts, ob)
        new_objects = connection.objects
        if len(new_objects) == 0:
            raise RuntimeError('Could not connect to %r' % connection.fullname)
        self._connect_and_disconnect(old_objects, new_objects, connection.force)

    def _connect_and_disconnect(self, old_objects, new_objects, force=False):
        if False:
            for i in range(10):
                print('nop')
        ' Update connections by disconnecting old and connecting new,\n        but try to keep connections that do not change.\n        '
        should_stay = {}
        i1 = 0
        while i1 < len(new_objects) and i1 < len(old_objects) and (new_objects[i1][0] is old_objects[i1][0]) and (new_objects[i1][1] == old_objects[i1][1]):
            should_stay[new_objects[i1][0].id + '-' + new_objects[i1][1]] = True
            i1 += 1
        (i2, i3) = (len(new_objects) - 1, len(old_objects) - 1)
        while i2 >= i1 and i3 >= i1 and (new_objects[i2][0] is old_objects[i3][0]) and (new_objects[i2][1] == old_objects[i3][1]):
            should_stay[new_objects[i2][0].id + '-' + new_objects[i2][1]] = True
            i2 -= 1
            i3 -= 1
        for i in range(i1, i3 + 1):
            (ob, type) = old_objects[i]
            if should_stay.get(ob.id + '-' + type, False) is False:
                ob.disconnect(type, self)
        for i in range(i1, i2 + 1):
            (ob, type) = new_objects[i]
            ob._register_reaction(type, self, force)

    def _seek_event_object(self, index, path, ob):
        if False:
            return 10
        ' Seek an event object based on the name (PScript compatible).\n        The path is a list: the path to the event, the last element being the\n        event type.\n        '
        connection = self._connections[index]
        if ob is None or len(path) == 0:
            return
        if len(path) == 1:
            if hasattr(ob, '_IS_COMPONENT'):
                connection.objects.append((ob, connection.type))
            if not path[0].endswith('**'):
                return
        (obname_full, path) = (path[0], path[1:])
        obname = obname_full.rstrip('*')
        selector = obname_full[len(obname):]
        if selector == '***':
            self._seek_event_object(index, path, ob)
        if hasattr(ob, '_IS_COMPONENT') and obname in ob.__properties__:
            name_label = obname + ':reconnect_' + str(index)
            connection.objects.append((ob, name_label))
            new_ob = getattr(ob, obname, None)
        else:
            new_ob = getattr(ob, obname, None)
        if len(selector) and selector in '***' and isinstance(new_ob, (tuple, list)):
            if len(selector) > 1:
                path = [obname + '***'] + path
            for isub in range(len(new_ob)):
                self._seek_event_object(index, path, new_ob[isub])
            return
        elif selector == '*':
            t = 'Invalid connection {name_full} because {name} is not a tuple/list.'
            raise RuntimeError(t.replace('{name_full}', obname_full).replace('{name}', obname))
        else:
            return self._seek_event_object(index, path, new_ob)