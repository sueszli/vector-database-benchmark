"""
:maintainer: Evan Borgstrom <evan@borgstrom.ca>

Pythonic object interface to creating state data, see the pyobjects renderer
for more documentation.
"""
import inspect
import logging
from salt.utils.odict import OrderedDict
REQUISITES = ('listen', 'onchanges', 'onfail', 'require', 'watch', 'use', 'listen_in', 'onchanges_in', 'onfail_in', 'require_in', 'watch_in', 'use_in')
log = logging.getLogger(__name__)

class StateException(Exception):
    pass

class DuplicateState(StateException):
    pass

class InvalidFunction(StateException):
    pass

class Registry:
    """
    The StateRegistry holds all of the states that have been created.
    """
    states = OrderedDict()
    requisites = []
    includes = []
    extends = OrderedDict()
    enabled = True

    @classmethod
    def empty(cls):
        if False:
            i = 10
            return i + 15
        cls.states = OrderedDict()
        cls.requisites = []
        cls.includes = []
        cls.extends = OrderedDict()

    @classmethod
    def include(cls, *args):
        if False:
            return 10
        if not cls.enabled:
            return
        cls.includes += args

    @classmethod
    def salt_data(cls):
        if False:
            for i in range(10):
                print('nop')
        states = OrderedDict([(id_, states_) for (id_, states_) in cls.states.items()])
        if cls.includes:
            states['include'] = cls.includes
        if cls.extends:
            states['extend'] = OrderedDict([(id_, states_) for (id_, states_) in cls.extends.items()])
        cls.empty()
        return states

    @classmethod
    def add(cls, id_, state, extend=False):
        if False:
            print('Hello World!')
        if not cls.enabled:
            return
        if extend:
            attr = cls.extends
        else:
            attr = cls.states
        if id_ in attr:
            if state.full_func in attr[id_]:
                raise DuplicateState("A state with id ''{}'', type ''{}'' exists".format(id_, state.full_func))
        else:
            attr[id_] = OrderedDict()
        if cls.requisites:
            for req in cls.requisites:
                if req.requisite not in state.kwargs:
                    state.kwargs[req.requisite] = []
                state.kwargs[req.requisite].append(req())
        attr[id_].update(state())

    @classmethod
    def extend(cls, id_, state):
        if False:
            print('Hello World!')
        cls.add(id_, state, extend=True)

    @classmethod
    def make_extend(cls, name):
        if False:
            return 10
        return StateExtend(name)

    @classmethod
    def push_requisite(cls, requisite):
        if False:
            return 10
        if not cls.enabled:
            return
        cls.requisites.append(requisite)

    @classmethod
    def pop_requisite(cls):
        if False:
            return 10
        if not cls.enabled:
            return
        del cls.requisites[-1]

class StateExtend:

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name

class StateRequisite:

    def __init__(self, requisite, module, id_):
        if False:
            for i in range(10):
                print('nop')
        self.requisite = requisite
        self.module = module
        self.id_ = id_

    def __call__(self):
        if False:
            return 10
        return {self.module: self.id_}

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        Registry.push_requisite(self)

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        Registry.pop_requisite()

class StateFactory:
    """
    The StateFactory is used to generate new States through a natural syntax

    It is used by initializing it with the name of the salt module::

        File = StateFactory("file")

    Any attribute accessed on the instance returned by StateFactory is a lambda
    that is a short cut for generating State objects::

        File.managed('/path/', owner='root', group='root')

    The kwargs are passed through to the State object
    """

    def __init__(self, module, valid_funcs=None):
        if False:
            i = 10
            return i + 15
        self.module = module
        if valid_funcs is None:
            valid_funcs = []
        self.valid_funcs = valid_funcs

    def __getattr__(self, func):
        if False:
            for i in range(10):
                print('nop')
        if self.valid_funcs and func not in self.valid_funcs:
            raise InvalidFunction("The function '{}' does not exist in the StateFactory for '{}'".format(func, self.module))

        def make_state(id_, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return State(id_, self.module, func, **kwargs)
        return make_state

    def __call__(self, id_, requisite='require'):
        if False:
            print('Hello World!')
        '\n        When an object is called it is being used as a requisite\n        '
        return StateRequisite(requisite, self.module, id_)

class State:
    """
    This represents a single item in the state tree

    The id_ is the id of the state, the func is the full name of the salt
    state (i.e. file.managed). All the keyword args you pass in become the
    properties of your state.
    """

    def __init__(self, id_, module, func, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.id_ = id_
        self.module = module
        self.func = func
        for attr in REQUISITES:
            if attr in kwargs:
                try:
                    iter(kwargs[attr])
                except TypeError:
                    kwargs[attr] = [kwargs[attr]]
        self.kwargs = kwargs
        if isinstance(self.id_, StateExtend):
            Registry.extend(self.id_.name, self)
            self.id_ = self.id_.name
        else:
            Registry.add(self.id_, self)
        self.requisite = StateRequisite('require', self.module, self.id_)

    @property
    def attrs(self):
        if False:
            for i in range(10):
                print('nop')
        kwargs = self.kwargs
        for attr in REQUISITES:
            if attr in kwargs:
                kwargs[attr] = [req() if isinstance(req, StateRequisite) else req for req in kwargs[attr]]
        return [{k: kwargs[k]} for k in sorted(kwargs.keys())]

    @property
    def full_func(self):
        if False:
            for i in range(10):
                print('nop')
        return '{!s}.{!s}'.format(self.module, self.func)

    def __str__(self):
        if False:
            print('Hello World!')
        return '{!s} = {!s}:{!s}'.format(self.id_, self.full_func, self.attrs)

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        return {self.full_func: self.attrs}

    def __enter__(self):
        if False:
            print('Hello World!')
        Registry.push_requisite(self.requisite)

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        Registry.pop_requisite()

class SaltObject:
    """
    Object based interface to the functions in __salt__

    .. code-block:: python
       :linenos:

        Salt = SaltObject(__salt__)
        Salt.cmd.run(bar)
    """

    def __init__(self, salt):
        if False:
            print('Hello World!')
        self._salt = salt

    def __getattr__(self, mod):
        if False:
            while True:
                i = 10

        class __wrapper__:

            def __getattr__(wself, func):
                if False:
                    i = 10
                    return i + 15
                try:
                    return self._salt['{}.{}'.format(mod, func)]
                except KeyError:
                    raise AttributeError
        return __wrapper__()

class MapMeta(type):
    """
    This is the metaclass for our Map class, used for building data maps based
    off of grain data.
    """

    @classmethod
    def __prepare__(metacls, name, bases):
        if False:
            return 10
        return OrderedDict()

    def __new__(cls, name, bases, attrs):
        if False:
            return 10
        c = type.__new__(cls, name, bases, attrs)
        c.__ordered_attrs__ = attrs.keys()
        return c

    def __init__(cls, name, bases, nmspc):
        if False:
            for i in range(10):
                print('nop')
        cls.__set_attributes__()
        super().__init__(name, bases, nmspc)

    def __set_attributes__(cls):
        if False:
            return 10
        match_info = []
        grain_targets = set()
        for item in cls.__ordered_attrs__:
            if item[0] == '_':
                continue
            filt = cls.__dict__[item]
            if not inspect.isclass(filt):
                continue
            grain = getattr(filt, '__grain__', 'os_family')
            grain_targets.add(grain)
            match = getattr(filt, '__match__', item)
            match_attrs = {}
            for name in filt.__dict__:
                if name[0] != '_':
                    match_attrs[name] = filt.__dict__[name]
            match_info.append((grain, match, match_attrs))
        try:
            if not hasattr(cls.priority, '__iter__'):
                log.error('pyobjects: priority must be an iterable')
            else:
                new_match_info = []
                for grain in cls.priority:
                    for (index, item) in list(enumerate(match_info)):
                        try:
                            if item[0] == grain:
                                new_match_info.append(item)
                                match_info[index] = None
                        except TypeError:
                            pass
                new_match_info.extend([x for x in match_info if x is not None])
                match_info = new_match_info
        except AttributeError:
            pass
        attrs = {}
        if match_info:
            grain_vals = Map.__salt__['grains.item'](*grain_targets)
            for (grain, match, match_attrs) in match_info:
                if grain not in grain_vals:
                    continue
                if grain_vals[grain] == match:
                    attrs.update(match_attrs)
        if hasattr(cls, 'merge'):
            pillar = Map.__salt__['pillar.get'](cls.merge)
            if pillar:
                attrs.update(pillar)
        for name in attrs:
            setattr(cls, name, attrs[name])

def need_salt(*a, **k):
    if False:
        return 10
    log.error('Map needs __salt__ set before it can be used!')
    return {}

class Map(metaclass=MapMeta):
    __salt__ = {'grains.filter_by': need_salt, 'pillar.get': need_salt}