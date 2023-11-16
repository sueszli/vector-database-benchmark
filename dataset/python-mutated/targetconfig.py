"""
This module contains utils for manipulating target configurations such as
compiler flags.
"""
import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils

class Option:
    """An option to be used in ``TargetConfig``.
    """
    __slots__ = ('_type', '_default', '_doc')

    def __init__(self, type, *, default, doc):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        type :\n            Type of the option value. It can be a callable.\n            The setter always calls ``self._type(value)``.\n        default :\n            The default value for the option.\n        doc : str\n            Docstring for the option.\n        '
        self._type = type
        self._default = default
        self._doc = doc

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        return self._type

    @property
    def default(self):
        if False:
            while True:
                i = 10
        return self._default

    @property
    def doc(self):
        if False:
            while True:
                i = 10
        return self._doc

class _FlagsStack(utils.ThreadLocalStack, stack_name='flags'):
    pass

class ConfigStack:
    """A stack for tracking target configurations in the compiler.

    It stores the stack in a thread-local class attribute. All instances in the
    same thread will see the same stack.
    """

    @classmethod
    def top_or_none(cls):
        if False:
            for i in range(10):
                print('nop')
        'Get the TOS or return None if no config is set.\n        '
        self = cls()
        if self:
            flags = self.top()
        else:
            flags = None
        return flags

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._stk = _FlagsStack()

    def top(self):
        if False:
            print('Hello World!')
        return self._stk.top()

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._stk)

    def enter(self, flags):
        if False:
            print('Hello World!')
        'Returns a contextmanager that performs ``push(flags)`` on enter and\n        ``pop()`` on exit.\n        '
        return self._stk.enter(flags)

class _MetaTargetConfig(type):
    """Metaclass for ``TargetConfig``.

    When a subclass of ``TargetConfig`` is created, all ``Option`` defined
    as class members will be parsed and corresponding getters, setters, and
    delters will be inserted.
    """

    def __init__(cls, name, bases, dct):
        if False:
            while True:
                i = 10
        'Invoked when subclass is created.\n\n        Insert properties for each ``Option`` that are class members.\n        All the options will be grouped inside the ``.options`` class\n        attribute.\n        '
        opts = {}
        for base_cls in reversed(bases):
            opts.update(base_cls.options)
        opts.update(cls.find_options(dct))
        cls.options = MappingProxyType(opts)

        def make_prop(name, option):
            if False:
                print('Hello World!')

            def getter(self):
                if False:
                    while True:
                        i = 10
                return self._values.get(name, option.default)

            def setter(self, val):
                if False:
                    return 10
                self._values[name] = option.type(val)

            def delter(self):
                if False:
                    print('Hello World!')
                del self._values[name]
            return property(getter, setter, delter, option.doc)
        for (name, option) in cls.options.items():
            setattr(cls, name, make_prop(name, option))

    def find_options(cls, dct):
        if False:
            while True:
                i = 10
        'Returns a new dict with all the items that are a mapping to an\n        ``Option``.\n        '
        return {k: v for (k, v) in dct.items() if isinstance(v, Option)}

class _NotSetType:

    def __repr__(self):
        if False:
            return 10
        return '<NotSet>'
_NotSet = _NotSetType()

class TargetConfig(metaclass=_MetaTargetConfig):
    """Base class for ``TargetConfig``.

    Subclass should fill class members with ``Option``. For example:

    >>> class MyTargetConfig(TargetConfig):
    >>>     a_bool_option = Option(type=bool, default=False, doc="a bool")
    >>>     an_int_option = Option(type=int, default=0, doc="an int")

    The metaclass will insert properties for each ``Option``. For example:

    >>> tc = MyTargetConfig()
    >>> tc.a_bool_option = True  # invokes the setter
    >>> print(tc.an_int_option)  # print the default
    """
    _ZLIB_CONFIG = {'wbits': -15}

    def __init__(self, copy_from=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        copy_from : TargetConfig or None\n            if None, creates an empty ``TargetConfig``.\n            Otherwise, creates a copy.\n        '
        self._values = {}
        if copy_from is not None:
            assert isinstance(copy_from, TargetConfig)
            self._values.update(copy_from._values)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        args = []
        defs = []
        for k in self.options:
            msg = f'{k}={getattr(self, k)}'
            if not self.is_set(k):
                defs.append(msg)
            else:
                args.append(msg)
        clsname = self.__class__.__name__
        return f"{clsname}({', '.join(args)}, [{', '.join(defs)}])"

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(tuple(sorted(self.values())))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, TargetConfig):
            return self.values() == other.values()
        else:
            return NotImplemented

    def values(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict of all the values\n        '
        return {k: getattr(self, k) for k in self.options}

    def is_set(self, name):
        if False:
            print('Hello World!')
        'Is the option set?\n        '
        self._guard_option(name)
        return name in self._values

    def discard(self, name):
        if False:
            while True:
                i = 10
        'Remove the option by name if it is defined.\n\n        After this, the value for the option will be set to its default value.\n        '
        self._guard_option(name)
        self._values.pop(name, None)

    def inherit_if_not_set(self, name, default=_NotSet):
        if False:
            print('Hello World!')
        'Inherit flag from ``ConfigStack``.\n\n        Parameters\n        ----------\n        name : str\n            Option name.\n        default : optional\n            When given, it overrides the default value.\n            It is only used when the flag is not defined locally and there is\n            no entry in the ``ConfigStack``.\n        '
        self._guard_option(name)
        if not self.is_set(name):
            cstk = ConfigStack()
            if cstk:
                top = cstk.top()
                setattr(self, name, getattr(top, name))
            elif default is not _NotSet:
                setattr(self, name, default)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Clone this instance.\n        '
        return type(self)(self)

    def summary(self) -> str:
        if False:
            while True:
                i = 10
        'Returns a ``str`` that summarizes this instance.\n\n        In contrast to ``__repr__``, only options that are explicitly set will\n        be shown.\n        '
        args = [f'{k}={v}' for (k, v) in self._summary_args()]
        clsname = self.__class__.__name__
        return f"{clsname}({', '.join(args)})"

    def _guard_option(self, name):
        if False:
            return 10
        if name not in self.options:
            msg = f'{name!r} is not a valid option for {type(self)}'
            raise ValueError(msg)

    def _summary_args(self):
        if False:
            return 10
        'returns a sorted sequence of 2-tuple containing the\n        ``(flag_name, flag_value)`` for flag that are set with a non-default\n        value.\n        '
        args = []
        for k in sorted(self.options):
            opt = self.options[k]
            if self.is_set(k):
                flagval = getattr(self, k)
                if opt.default != flagval:
                    v = (k, flagval)
                    args.append(v)
        return args

    @classmethod
    def _make_compression_dictionary(cls) -> bytes:
        if False:
            print('Hello World!')
        'Returns a ``bytes`` object suitable for use as a dictionary for\n        compression.\n        '
        buf = []
        buf.append('numba')
        buf.append(cls.__class__.__name__)
        buf.extend(['True', 'False'])
        for (k, opt) in cls.options.items():
            buf.append(k)
            buf.append(str(opt.default))
        return ''.join(buf).encode()

    def get_mangle_string(self) -> str:
        if False:
            while True:
                i = 10
        'Return a string suitable for symbol mangling.\n        '
        zdict = self._make_compression_dictionary()
        comp = zlib.compressobj(zdict=zdict, level=zlib.Z_BEST_COMPRESSION, **self._ZLIB_CONFIG)
        buf = [comp.compress(self.summary().encode())]
        buf.append(comp.flush())
        return base64.b64encode(b''.join(buf)).decode()

    @classmethod
    def demangle(cls, mangled: str) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the demangled result from ``.get_mangle_string()``\n        '

        def repl(x):
            if False:
                print('Hello World!')
            return chr(int('0x' + x.group(0)[1:], 16))
        unescaped = re.sub('_[a-zA-Z0-9][a-zA-Z0-9]', repl, mangled)
        raw = base64.b64decode(unescaped)
        zdict = cls._make_compression_dictionary()
        dc = zlib.decompressobj(zdict=zdict, **cls._ZLIB_CONFIG)
        buf = []
        while raw:
            buf.append(dc.decompress(raw))
            raw = dc.unconsumed_tail
        buf.append(dc.flush())
        return b''.join(buf).decode()