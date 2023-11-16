"""
Implements the Component class; the core class that has properties,
actions that mutate the properties, and reactions that react to the
events and changes in properties.
"""
import sys
from ._dict import Dict
from ._attribute import Attribute
from ._action import ActionDescriptor, Action
from ._reaction import ReactionDescriptor, Reaction, looks_like_method
from ._property import Property
from ._emitter import EmitterDescriptor
from ._loop import loop, this_is_js
from . import logger
setTimeout = console = None

def with_metaclass(meta, *bases):
    if False:
        for i in range(10):
            print('nop')
    'Create a base class with a metaclass.'
    tmp_name = b'tmp_class' if sys.version_info[0] == 2 else 'tmp_class'

    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            if False:
                return 10
            return meta(name, bases, d)
    return type.__new__(metaclass, tmp_name, (), {})

def new_type(name, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    ' Alternative for type(...) to be legacy-py compatible.\n    '
    name = name.encode() if sys.version_info[0] == 2 else name
    return type(name, *args, **kwargs)

class ComponentMeta(type):
    """ Meta class for Component
    * Set the name of property desciptors.
    * Set __actions__, __reactions__, __emitters__ and __properties__ class attributes.
    * Create private methods (e.g. mutator functions and prop validators).
    """

    def __init__(cls, name, bases, dct):
        if False:
            for i in range(10):
                print('nop')
        cls._finish_properties(dct)
        cls._init_hook1(name, bases, dct)
        cls._set_summaries()
        cls._init_hook2(name, bases, dct)
        type.__init__(cls, name, bases, dct)

    def _init_hook1(cls, name, bases, dct):
        if False:
            i = 10
            return i + 15
        ' Overloaded in flexx.app.AppComponentMeta.\n        '
        pass

    def _init_hook2(cls, name, bases, dct):
        if False:
            i = 10
            return i + 15
        ' Overloaded in flexx.app.AppComponentMeta.\n        '
        pass

    def _set_cls_attr(cls, dct, name, att):
        if False:
            return 10
        dct[name] = att
        setattr(cls, name, att)

    def _finish_properties(cls, dct):
        if False:
            return 10
        ' Finish properties:\n\n        * Create a mutator function for convenience.\n        * Create validator function.\n        * If needed, create a corresponding set_xx action.\n        '
        for name in list(dct.keys()):
            if name.startswith('__'):
                continue
            val = getattr(cls, name)
            if isinstance(val, type) and issubclass(val, (Attribute, Property)):
                raise TypeError('Attributes and Properties should be instantiated, use ``foo = IntProp()`` instead of ``foo = IntProp``.')
            elif isinstance(val, Attribute):
                val._set_name(name)
            elif isinstance(val, Property):
                val._set_name(name)
                cls._set_cls_attr(dct, '_' + name + '_validate', val._validate_py)
                cls._set_cls_attr(dct, '_mutate_' + name, val.make_mutator())
                action_name = ('_set' if name.startswith('_') else 'set_') + name
                if val._settable and (not hasattr(cls, action_name)):
                    action_des = ActionDescriptor(val.make_set_action(), action_name, 'Setter for the %r property.' % name)
                    cls._set_cls_attr(dct, action_name, action_des)

    def _set_summaries(cls):
        if False:
            for i in range(10):
                print('nop')
        ' Analyse the class and set lists __actions__, __emitters__,\n        __properties__, and __reactions__.\n        '
        attributes = {}
        properties = {}
        actions = {}
        emitters = {}
        reactions = {}
        for name in dir(cls):
            if name.startswith('__'):
                continue
            val = getattr(cls, name)
            if isinstance(val, Attribute):
                attributes[name] = val
            elif isinstance(val, Property):
                properties[name] = val
            elif isinstance(val, ActionDescriptor):
                actions[name] = val
            elif isinstance(val, ReactionDescriptor):
                reactions[name] = val
            elif isinstance(val, EmitterDescriptor):
                emitters[name] = val
            elif isinstance(val, (Action, Reaction)):
                raise RuntimeError('Class methods can only be made actions or reactions using the corresponding decorators (%r)' % name)
        cls.__attributes__ = [name for name in sorted(attributes.keys())]
        cls.__properties__ = [name for name in sorted(properties.keys())]
        cls.__actions__ = [name for name in sorted(actions.keys())]
        cls.__emitters__ = [name for name in sorted(emitters.keys())]
        cls.__reactions__ = [name for name in sorted(reactions.keys())]

class Component(with_metaclass(ComponentMeta, object)):
    """ The base component class.

    Components have attributes to represent static values, properties
    to represent state, actions that can mutate properties, and
    reactions that react to events such as property changes.

    Initial values of properties can be provided by passing them
    as keyword arguments.

    Subclasses can use :class:`Property <flexx.event.Property>` (or one
    of its subclasses) to define properties, and the
    :func:`action <flexx.event.action>`, :func:`reaction <flexx.event.reaction>`,
    and :func:`emitter <flexx.event.emitter>` decorators to create actions,
    reactions. and emitters, respectively.

    .. code-block:: python

        class MyComponent(event.Component):

            foo = event.FloatProp(7, settable=True)
            spam = event.Attribute()

            @event.action
            def inrease_foo(self):
                self._mutate_foo(self.foo + 1)

            @event.reaction('foo')
            def on_foo(self, *events):
                print('foo was set to', self.foo)

            @event.reaction('bar')
            def on_bar(self, *events):
                for ev in events:
                    print('bar event was emitted')

            @event.emitter
            def bar(self, v):
                return dict(value=v)  # the event to emit

    """
    _IS_COMPONENT = True
    _COUNT = 0
    id = Attribute(doc='The string by which this component is identified.')

    def __init__(self, *init_args, **property_values):
        if False:
            for i in range(10):
                print('nop')
        Component._COUNT += 1
        self._id = self.__class__.__name__ + str(Component._COUNT)
        self._disposed = False
        self.__handlers = {}
        self.__pending_events = []
        self.__anonymous_reactions = []
        self.__initial_mutation = False
        for name in self.__emitters__:
            self.__handlers.setdefault(name, [])
        for name in self.__properties__:
            self.__handlers.setdefault(name, [])
        with self:
            self._comp_init_property_values(property_values)
            self.init(*init_args)
        self._comp_init_reactions()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return "<Component '%s' at 0x%x>" % (self._id, id(self))

    def _comp_init_property_values(self, property_values):
        if False:
            i = 10
            return i + 15
        ' Initialize property values, combining given kwargs (in order)\n        and default values.\n        Property values are popped when consumed so that the remainer is used for\n        other initialisations without mixup.\n        '
        values = []
        for name in self.__properties__:
            prop = getattr(self.__class__, name)
            setattr(self, '_' + name + '_value', prop._default)
            if name not in property_values:
                values.append((name, prop._default))
        for (name, value) in list(property_values.items()):
            if name not in self.__properties__:
                if name in self.__attributes__:
                    raise AttributeError('%s.%s is an attribute, not a property' % (self._id, name))
                elif self._has_proxy is True:
                    raise AttributeError('%s does not have property %s.' % (self._id, name))
            if callable(value):
                self._comp_make_implicit_setter(name, value)
                property_values.pop(name)
                continue
            if name in self.__properties__:
                values.append((name, value))
                property_values.pop(name)
        self._comp_apply_property_values(values)

    def _comp_apply_property_values(self, values):
        if False:
            while True:
                i = 10
        ' Apply given property values, prefer using a setter, mutate otherwise.\n        '
        self.__initial_mutation = True
        for (name, value) in values:
            self._mutate(name, value)
        for (name, value) in values:
            setter_name = ('_set' if name.startswith('_') else 'set_') + name
            setter = getattr(self, setter_name, None)
            if setter is not None:
                if getattr(setter, 'is_autogenerated', None) is False:
                    setter(value)
        self.__initial_mutation = False

    def _comp_make_implicit_setter(self, prop_name, func):
        if False:
            print('Hello World!')
        setter_func = getattr(self, 'set_' + prop_name, None)
        if setter_func is None:
            t = '%s does not have a set_%s() action for property %s.'
            raise TypeError(t % (self._id, prop_name, prop_name))
        setter_reaction = lambda : setter_func(func())
        reaction = Reaction(self, setter_reaction, 'auto', [])
        self.__anonymous_reactions.append(reaction)

    def _comp_init_reactions(self):
        if False:
            i = 10
            return i + 15
        ' Create our own reactions. These will immediately connect.\n        '
        if self.__pending_events is not None:
            self.__pending_events.append(None)
            loop.call_soon(self._comp_stop_capturing_events)
        for name in self.__reactions__:
            reaction = getattr(self, name)
            if reaction.get_mode() == 'auto':
                ev = Dict(source=self, type='', label='')
                loop.add_reaction_event(reaction, ev)
        for reaction in self.__anonymous_reactions:
            if reaction.get_mode() == 'auto':
                ev = Dict(source=self, type='', label='')
                loop.add_reaction_event(reaction, ev)

    def _comp_stop_capturing_events(self):
        if False:
            return 10
        ' Stop capturing events and flush the captured events.\n        This gets scheduled to be called asap after initialization. But\n        components created in our init() go first.\n        '
        events = self.__pending_events
        self.__pending_events = None
        allow_reconnect = False
        for ev in events:
            if ev is None:
                allow_reconnect = True
                continue
            ev.allow_reconnect = allow_reconnect
            self.emit(ev.type, ev)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        loop._activate_component(self)
        loop.call_soon(self.__check_not_active)
        return self

    def __exit__(self, type, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        loop._deactivate_component(self)

    def __check_not_active(self):
        if False:
            while True:
                i = 10
        active_components = loop.get_active_components()
        if self in active_components:
            raise RuntimeError('It seems that the event loop is processing events while a Component is active. This has a high risk on race conditions.')

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        ' Initializer method. This method can be overloaded when\n        creating a custom class. It is called with this component as a\n        context manager (i.e. it is the active component), and it receives\n        any positional arguments that were passed to the constructor.\n        '
        pass

    def __del__(self):
        if False:
            print('Hello World!')
        if not self._disposed:
            self._dispose()

    def dispose(self):
        if False:
            return 10
        ' Use this to dispose of the object to prevent memory leaks.\n        Make all subscribed reactions forget about this object, clear\n        all references to subscribed reactions, and disconnect all reactions\n        defined on this object.\n        '
        self._dispose()

    def _dispose(self):
        if False:
            print('Hello World!')
        self._disposed = True
        if not this_is_js():
            logger.debug('Disposing Component %r' % self)
        for (name, reactions) in self.__handlers.items():
            for i in range(len(reactions)):
                reactions[i][1]._clear_component_refs(self)
            while len(reactions):
                reactions.pop()
        for i in range(len(self.__reactions__)):
            getattr(self, self.__reactions__[i]).dispose()

    def _registered_reactions_hook(self):
        if False:
            return 10
        ' This method is called when the reactions change, can be overloaded\n        in subclasses. The original method returns a list of event types for\n        which there is at least one registered reaction. Overloaded methods\n        should return this list too.\n        '
        used_event_types = []
        for (key, reactions) in self.__handlers.items():
            if len(reactions) > 0:
                used_event_types.append(key)
        return used_event_types

    def _register_reaction(self, event_type, reaction, force=False):
        if False:
            print('Hello World!')
        (type, _, label) = event_type.partition(':')
        label = label or reaction._name
        reactions = self.__handlers.get(type, None)
        if reactions is None:
            reactions = []
            self.__handlers[type] = reactions
            if force:
                pass
            elif type.startswith('mouse_'):
                t = 'The event "{}" has been renamed to "pointer{}".'
                logger.warning(t.format(type, type[5:]))
            else:
                msg = 'Event type "{type}" does not exist on component {id}. ' + 'Use "!{type}" or "!xx.yy.{type}" to suppress this warning.'
                msg = msg.replace('{type}', type).replace('{id}', self._id)
                logger.warning(msg)
        comp1 = label + '-' + reaction._id
        for i in range(len(reactions)):
            comp2 = reactions[i][0] + '-' + reactions[i][1]._id
            if comp1 < comp2:
                reactions.insert(i, (label, reaction))
                break
            elif comp1 == comp2:
                break
        else:
            reactions.append((label, reaction))
        self._registered_reactions_hook()

    def disconnect(self, type, reaction=None):
        if False:
            for i in range(10):
                print('nop')
        ' Disconnect reactions.\n\n        Parameters:\n            type (str): the type for which to disconnect any reactions.\n                Can include the label to only disconnect reactions that\n                were registered with that label.\n            reaction (optional): the reaction object to disconnect. If given,\n               only this reaction is removed.\n        '
        (type, _, label) = type.partition(':')
        reactions = self.__handlers.get(type, ())
        for i in range(len(reactions) - 1, -1, -1):
            entry = reactions[i]
            if not (label and label != entry[0] or (reaction and reaction is not entry[1])):
                reactions.pop(i)
        self._registered_reactions_hook()

    def emit(self, type, info=None):
        if False:
            i = 10
            return i + 15
        ' Generate a new event and dispatch to all event reactions.\n\n        Arguments:\n            type (str): the type of the event. Should not include a label.\n            info (dict): Optional. Additional information to attach to\n                the event object. Note that the actual event is a Dict object\n                that allows its elements to be accesses as attributes.\n        '
        info = {} if info is None else info
        (type, _, label) = type.partition(':')
        if len(label):
            raise ValueError('The type given to emit() should not include a label.')
        if not isinstance(info, dict):
            raise TypeError('Info object (for %r) must be a dict, not %r' % (type, info))
        ev = Dict(info)
        ev.type = type
        ev.source = self
        if self.__pending_events is not None:
            self.__pending_events.append(ev)
        else:
            reactions = self.__handlers.get(ev.type, ())
            for i in range(len(reactions)):
                (label, reaction) = reactions[i]
                if label.startswith('reconnect_'):
                    if getattr(ev, 'allow_reconnect', True) is True:
                        index = int(label.split('_')[1])
                        reaction.reconnect(index)
                else:
                    loop.add_reaction_event(reaction, ev)
        return ev

    def _mutate(self, prop_name, value, mutation='set', index=-1):
        if False:
            i = 10
            return i + 15
        " Mutate a :class:`property <flexx.event.Property>`.\n        Can only be called from an :class:`action <flexx.event.action>`.\n\n        Each Component class will also have an auto-generated mutator function\n        for each property: e.g. property ``foo`` can be mutated with\n        ``c._mutate('foo', ..)`` or ``c._mutate_foo(..)``.\n\n        Arguments:\n            prop_name (str): the name of the property being mutated.\n            value: the new value, or the partial value for partial mutations.\n            mutation (str): the kind of mutation to apply. Default is 'set'.\n               Partial mutations to list-like\n               :class:`properties <flexx.event.Property>` can be applied by using\n               'insert', 'remove', or 'replace'. If other than 'set', index must\n               be specified, and >= 0. If 'remove', then value must be an int\n               specifying the number of items to remove.\n            index: the index at which to insert, remove or replace items. Must\n                be an int for list properties.\n\n        The 'replace' mutation also supports multidensional (numpy) arrays.\n        In this case ``value`` can be an ndarray to patch the data with, and\n        ``index`` a tuple of elements.\n        "
        if not isinstance(prop_name, str):
            raise TypeError("_mutate's first arg must be str, not %s" % prop_name.__class__)
        if prop_name not in self.__properties__:
            cname = self.__class__.__name__
            raise AttributeError('%s object has no property %r' % (cname, prop_name))
        if loop.can_mutate(self) is False:
            raise AttributeError('Trying to mutate property %s outside of an action or context.' % prop_name)
        private_name = '_' + prop_name + '_value'
        validator_name = '_' + prop_name + '_validate'
        old = getattr(self, private_name)
        if mutation == 'set':
            value2 = getattr(self, validator_name)(value)
            setattr(self, private_name, value2)
            if this_is_js():
                is_equal = old == value2
            elif hasattr(old, 'dtype') and hasattr(value2, 'dtype'):
                import numpy as np
                is_equal = np.array_equal(old, value2)
            else:
                is_equal = type(old) == type(value2) and old == value2
            if self.__initial_mutation is True:
                old = value2
                is_equal = False
            if not is_equal:
                self.emit(prop_name, dict(new_value=value2, old_value=old, mutation=mutation))
                return True
        else:
            ev = Dict()
            ev.objects = value
            ev.mutation = mutation
            ev.index = index
            if isinstance(old, dict):
                if index != -1:
                    raise IndexError('For in-place dict mutations, the index is not used, and must be -1.')
                mutate_dict(old, ev)
            else:
                if index < 0:
                    raise IndexError('For insert, remove, and replace mutations, the index must be >= 0.')
                mutate_array(old, ev)
            self.emit(prop_name, ev)
            return True

    def get_event_types(self):
        if False:
            i = 10
            return i + 15
        ' Get the known event types for this component. Returns\n        a list of event type names, for which there is a\n        property/emitter or for which any reactions are registered.\n        Sorted alphabetically. Intended mostly for debugging purposes.\n        '
        types = list(self.__handlers)
        types.sort()
        return types

    def get_event_handlers(self, type):
        if False:
            print('Hello World!')
        ' Get a list of reactions for the given event type. The order\n        is the order in which events are handled: alphabetically by\n        label. Intended mostly for debugging purposes.\n\n        Parameters:\n            type (str): the type of event to get reactions for. Should not\n                include a label.\n\n        '
        if not type:
            raise TypeError('get_event_handlers() missing "type" argument.')
        (type, _, label) = type.partition(':')
        if len(label):
            raise ValueError('The type given to get_event_handlers() should not include a label.')
        reactions = self.__handlers.get(type, ())
        return [h[1] for h in reactions]

    def reaction(self, *connection_strings):
        if False:
            for i in range(10):
                print('nop')
        ' Create a reaction by connecting a function to one or more events of\n        this instance. Can also be used as a decorator. See the\n        :func:`reaction <flexx.event.reaction>` decorator, and the intro\n        docs for more information.\n        '
        mode = 'normal'
        if not connection_strings or (len(connection_strings) == 1 and callable(connection_strings[0])):
            raise RuntimeError('Component.reaction() needs one or more connection strings.')
        func = None
        if callable(connection_strings[0]):
            func = connection_strings[0]
            connection_strings = connection_strings[1:]
        elif callable(connection_strings[-1]):
            func = connection_strings[-1]
            connection_strings = connection_strings[:-1]
        for s in connection_strings:
            if not (isinstance(s, str) and len(s) > 0):
                raise ValueError('Connection string must be nonempty string.')

        def _react(func):
            if False:
                return 10
            if not callable(func):
                raise TypeError('Component.reaction() decorator requires a callable.')
            if looks_like_method(func):
                return ReactionDescriptor(func, mode, connection_strings, self)
            else:
                return Reaction(self, func, mode, connection_strings)
        if func is not None:
            return _react(func)
        else:
            return _react

def mutate_dict(d, ev):
    if False:
        print('Hello World!')
    " Function to mutate an dict property in-place.\n    Used by Component. The ``ev`` must be a dict with elements:\n\n    * mutation: 'set', 'insert', 'remove' or 'replace'.\n    * objects: the dict to set/insert/replace, or a list if keys to remove.\n    * index: not used.\n    "
    mutation = ev['mutation']
    objects = ev['objects']
    if mutation in ('set', 'insert', 'replace'):
        if mutation == 'set':
            d.clear()
        assert isinstance(objects, dict)
        for (key, val) in objects.items():
            d[key] = val
    elif mutation == 'remove':
        assert isinstance(objects, (tuple, list))
        for key in objects:
            d.pop(key)
    else:
        raise NotImplementedError(mutation)

def _mutate_array_py(array, ev):
    if False:
        print('Hello World!')
    " Function to mutate a list- or array-like property in-place.\n    Used by Component. The ``ev`` must be a dict with elements:\n\n    * mutation: 'set', 'insert', 'remove' or 'replace'.\n    * objects: the values to set/insert/replace, or the number of iterms to remove.\n    * index: the (non-negative) index to insert/replace/remove at.\n    "
    is_nd = hasattr(array, 'shape') and hasattr(array, 'dtype')
    mutation = ev['mutation']
    index = ev['index']
    objects = ev['objects']
    if is_nd:
        if mutation == 'set':
            raise NotImplementedError('Cannot set numpy array in-place')
        elif mutation in ('insert', 'remove'):
            raise NotImplementedError('Cannot resize numpy arrays')
        elif mutation == 'replace':
            if isinstance(index, tuple):
                slices = tuple((slice(index[i], index[i] + objects.shape[i], 1) for i in range(len(index))))
                array[slices] = objects
            else:
                array[index:index + len(objects)] = objects
    elif mutation == 'set':
        array[:] = objects
    elif mutation == 'insert':
        array[index:index] = objects
    elif mutation == 'remove':
        assert isinstance(objects, int)
        array[index:index + objects] = []
    elif mutation == 'replace':
        array[index:index + len(objects)] = objects
    else:
        raise NotImplementedError(mutation)

def _mutate_array_js(array, ev):
    if False:
        return 10
    ' Logic to mutate an list-like or array-like property in-place, in JS.\n    '
    is_nd = hasattr(array, 'shape') and hasattr(array, 'dtype')
    mutation = ev.mutation
    index = ev.index
    objects = ev.objects
    if is_nd is True:
        if mutation == 'set':
            raise NotImplementedError('Cannot set nd array in-place')
        elif mutation in ('extend', 'insert', 'remove'):
            raise NotImplementedError('Cannot resize nd arrays')
        elif mutation == 'replace':
            raise NotImplementedError('Cannot replace items in nd array')
    else:
        if mutation == 'remove':
            assert isinstance(objects, float)
        elif not isinstance(objects, list):
            raise TypeError('Inplace list/array mutating requires a list of objects.')
        if mutation == 'set':
            array.splice(0, len(array), *objects)
        elif mutation == 'insert':
            array.splice(index, 0, *objects)
        elif mutation == 'remove':
            array.splice(index, objects)
        elif mutation == 'replace':
            array.splice(index, len(objects), *objects)
        else:
            raise NotImplementedError(mutation)
mutate_array = _mutate_array_py
_mutate_dict_js = _mutate_dict_py = mutate_dict