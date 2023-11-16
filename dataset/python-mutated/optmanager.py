from __future__ import annotations
import contextlib
import copy
import os
import pprint
import textwrap
import weakref
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import TextIO
import ruamel.yaml
from mitmproxy import exceptions
from mitmproxy.utils import signals
from mitmproxy.utils import typecheck
'\n    The base implementation for Options.\n'
unset = object()

class _Option:
    __slots__ = ('name', 'typespec', 'value', '_default', 'choices', 'help')

    def __init__(self, name: str, typespec: type | object, default: Any, help: str, choices: Sequence[str] | None) -> None:
        if False:
            return 10
        typecheck.check_option_type(name, default, typespec)
        self.name = name
        self.typespec = typespec
        self._default = default
        self.value = unset
        self.help = textwrap.dedent(help).strip().replace('\n', ' ')
        self.choices = choices

    def __repr__(self):
        if False:
            return 10
        return f'{self.current()} [{self.typespec}]'

    @property
    def default(self):
        if False:
            print('Hello World!')
        return copy.deepcopy(self._default)

    def current(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if self.value is unset:
            v = self.default
        else:
            v = self.value
        return copy.deepcopy(v)

    def set(self, value: Any) -> None:
        if False:
            return 10
        typecheck.check_option_type(self.name, value, self.typespec)
        self.value = value

    def reset(self) -> None:
        if False:
            print('Hello World!')
        self.value = unset

    def has_changed(self) -> bool:
        if False:
            return 10
        return self.current() != self.default

    def __eq__(self, other) -> bool:
        if False:
            i = 10
            return i + 15
        for i in self.__slots__:
            if getattr(self, i) != getattr(other, i):
                return False
        return True

    def __deepcopy__(self, _):
        if False:
            print('Hello World!')
        o = _Option(self.name, self.typespec, self.default, self.help, self.choices)
        if self.has_changed():
            o.value = self.current()
        return o

@dataclass
class _UnconvertedStrings:
    val: list[str]

def _sig_changed_spec(updated: set[str]) -> None:
    if False:
        while True:
            i = 10
    ...

def _sig_errored_spec(exc: Exception) -> None:
    if False:
        while True:
            i = 10
    ...

class OptManager:
    """
    OptManager is the base class from which Options objects are derived.

    .changed is a Signal that triggers whenever options are
    updated. If any handler in the chain raises an exceptions.OptionsError
    exception, all changes are rolled back, the exception is suppressed,
    and the .errored signal is notified.

    Optmanager always returns a deep copy of options to ensure that
    mutation doesn't change the option state inadvertently.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.deferred: dict[str, Any] = {}
        self.changed = signals.SyncSignal(_sig_changed_spec)
        self.changed.connect(self._notify_subscribers)
        self.errored = signals.SyncSignal(_sig_errored_spec)
        self._subscriptions: list[tuple[weakref.ref[Callable], set[str]]] = []
        self._options: dict[str, Any] = {}

    def add_option(self, name: str, typespec: type | object, default: Any, help: str, choices: Sequence[str] | None=None) -> None:
        if False:
            return 10
        self._options[name] = _Option(name, typespec, default, help, choices)
        self.changed.send(updated={name})

    @contextlib.contextmanager
    def rollback(self, updated, reraise=False):
        if False:
            return 10
        old = copy.deepcopy(self._options)
        try:
            yield
        except exceptions.OptionsError as e:
            self.errored.send(exc=e)
            self.__dict__['_options'] = old
            self.changed.send(updated=updated)
            if reraise:
                raise e

    def subscribe(self, func, opts):
        if False:
            return 10
        '\n        Subscribe a callable to the .changed signal, but only for a\n        specified list of options. The callable should accept arguments\n        (options, updated), and may raise an OptionsError.\n\n        The event will automatically be unsubscribed if the callable goes out of scope.\n        '
        for i in opts:
            if i not in self._options:
                raise exceptions.OptionsError('No such option: %s' % i)
        self._subscriptions.append((signals.make_weak_ref(func), set(opts)))

    def _notify_subscribers(self, updated) -> None:
        if False:
            i = 10
            return i + 15
        cleanup = False
        for (ref, opts) in self._subscriptions:
            callback = ref()
            if callback is not None:
                if opts & updated:
                    callback(self, updated)
            else:
                cleanup = True
        if cleanup:
            self.__dict__['_subscriptions'] = [(ref, opts) for (ref, opts) in self._subscriptions if ref() is not None]

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, OptManager):
            return self._options == other._options
        return False

    def __deepcopy__(self, memodict=None):
        if False:
            return 10
        o = OptManager()
        o.__dict__['_options'] = copy.deepcopy(self._options, memodict)
        return o
    __copy__ = __deepcopy__

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        if attr in self._options:
            return self._options[attr].current()
        else:
            raise AttributeError('No such option: %s' % attr)

    def __setattr__(self, attr, value):
        if False:
            return 10
        opts = self.__dict__.get('_options')
        if not opts:
            super().__setattr__(attr, value)
        else:
            self.update(**{attr: value})

    def keys(self):
        if False:
            while True:
                i = 10
        return set(self._options.keys())

    def items(self):
        if False:
            i = 10
            return i + 15
        return self._options.items()

    def __contains__(self, k):
        if False:
            for i in range(10):
                print('nop')
        return k in self._options

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Restore defaults for all options.\n        '
        for o in self._options.values():
            o.reset()
        self.changed.send(updated=set(self._options.keys()))

    def update_known(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Update and set all known options from kwargs. Returns a dictionary\n        of unknown options.\n        '
        (known, unknown) = ({}, {})
        for (k, v) in kwargs.items():
            if k in self._options:
                known[k] = v
            else:
                unknown[k] = v
        updated = set(known.keys())
        if updated:
            with self.rollback(updated, reraise=True):
                for (k, v) in known.items():
                    self._options[k].set(v)
                self.changed.send(updated=updated)
        return unknown

    def update_defer(self, **kwargs):
        if False:
            return 10
        unknown = self.update_known(**kwargs)
        self.deferred.update(unknown)

    def update(self, **kwargs):
        if False:
            return 10
        u = self.update_known(**kwargs)
        if u:
            raise KeyError('Unknown options: %s' % ', '.join(u.keys()))

    def setter(self, attr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate a setter for a given attribute. This returns a callable\n        taking a single argument.\n        '
        if attr not in self._options:
            raise KeyError('No such option: %s' % attr)

        def setter(x):
            if False:
                return 10
            setattr(self, attr, x)
        return setter

    def toggler(self, attr):
        if False:
            return 10
        '\n        Generate a toggler for a boolean attribute. This returns a callable\n        that takes no arguments.\n        '
        if attr not in self._options:
            raise KeyError('No such option: %s' % attr)
        o = self._options[attr]
        if o.typespec != bool:
            raise ValueError('Toggler can only be used with boolean options')

        def toggle():
            if False:
                i = 10
                return i + 15
            setattr(self, attr, not getattr(self, attr))
        return toggle

    def default(self, option: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self._options[option].default

    def has_changed(self, option):
        if False:
            return 10
        '\n        Has the option changed from the default?\n        '
        return self._options[option].has_changed()

    def merge(self, opts):
        if False:
            while True:
                i = 10
        '\n        Merge a dict of options into this object. Options that have None\n        value are ignored. Lists and tuples are appended to the current\n        option value.\n        '
        toset = {}
        for (k, v) in opts.items():
            if v is not None:
                if isinstance(v, (list, tuple)):
                    toset[k] = getattr(self, k) + v
                else:
                    toset[k] = v
        self.update(**toset)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        options = pprint.pformat(self._options, indent=4).strip(' {}')
        if '\n' in options:
            options = '\n    ' + options + '\n'
        return '{mod}.{cls}({{{options}}})'.format(mod=type(self).__module__, cls=type(self).__name__, options=options)

    def set(self, *specs: str, defer: bool=False) -> None:
        if False:
            return 10
        '\n        Takes a list of set specification in standard form (option=value).\n        Options that are known are updated immediately. If defer is true,\n        options that are not known are deferred, and will be set once they\n        are added.\n\n        May raise an `OptionsError` if a value is malformed or an option is unknown and defer is False.\n        '
        unprocessed: dict[str, list[str]] = {}
        for spec in specs:
            if '=' in spec:
                (name, value) = spec.split('=', maxsplit=1)
                unprocessed.setdefault(name, []).append(value)
            else:
                unprocessed.setdefault(spec, [])
        processed: dict[str, Any] = {}
        for name in list(unprocessed.keys()):
            if name in self._options:
                processed[name] = self._parse_setval(self._options[name], unprocessed.pop(name))
        if defer:
            self.deferred.update({k: _UnconvertedStrings(v) for (k, v) in unprocessed.items()})
        elif unprocessed:
            raise exceptions.OptionsError(f"Unknown option(s): {', '.join(unprocessed)}")
        self.update(**processed)

    def process_deferred(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Processes options that were deferred in previous calls to set, and\n        have since been added.\n        '
        update: dict[str, Any] = {}
        for (optname, value) in self.deferred.items():
            if optname in self._options:
                if isinstance(value, _UnconvertedStrings):
                    value = self._parse_setval(self._options[optname], value.val)
                update[optname] = value
        self.update(**update)
        for k in update.keys():
            del self.deferred[k]

    def _parse_setval(self, o: _Option, values: list[str]) -> Any:
        if False:
            while True:
                i = 10
        '\n        Convert a string to a value appropriate for the option type.\n        '
        if o.typespec == Sequence[str]:
            return values
        if len(values) > 1:
            raise exceptions.OptionsError(f'Received multiple values for {o.name}: {values}')
        optstr: str | None
        if values:
            optstr = values[0]
        else:
            optstr = None
        if o.typespec in (str, Optional[str]):
            if o.typespec == str and optstr is None:
                raise exceptions.OptionsError(f'Option is required: {o.name}')
            return optstr
        elif o.typespec in (int, Optional[int]):
            if optstr:
                try:
                    return int(optstr)
                except ValueError:
                    raise exceptions.OptionsError(f'Not an integer: {optstr}')
            elif o.typespec == int:
                raise exceptions.OptionsError(f'Option is required: {o.name}')
            else:
                return None
        elif o.typespec == bool:
            if optstr == 'toggle':
                return not o.current()
            if not optstr or optstr == 'true':
                return True
            elif optstr == 'false':
                return False
            else:
                raise exceptions.OptionsError('Boolean must be "true", "false", or have the value omitted (a synonym for "true").')
        raise NotImplementedError(f'Unsupported option type: {o.typespec}')

    def make_parser(self, parser, optname, metavar=None, short=None):
        if False:
            i = 10
            return i + 15
        '\n        Auto-Create a command-line parser entry for a named option. If the\n        option does not exist, it is ignored.\n        '
        if optname not in self._options:
            return
        o = self._options[optname]

        def mkf(x, s):
            if False:
                return 10
            x = x.replace('_', '-')
            f = ['--%s' % x]
            if s:
                f.append('-' + s)
            return f
        flags = mkf(optname, short)
        if o.typespec == bool:
            g = parser.add_mutually_exclusive_group(required=False)
            onf = mkf(optname, None)
            offf = mkf('no-' + optname, None)
            if short:
                if o.default:
                    offf = mkf('no-' + optname, short)
                else:
                    onf = mkf(optname, short)
            g.add_argument(*offf, action='store_false', dest=optname)
            g.add_argument(*onf, action='store_true', dest=optname, help=o.help)
            parser.set_defaults(**{optname: None})
        elif o.typespec in (int, Optional[int]):
            parser.add_argument(*flags, action='store', type=int, dest=optname, help=o.help, metavar=metavar)
        elif o.typespec in (str, Optional[str]):
            parser.add_argument(*flags, action='store', type=str, dest=optname, help=o.help, metavar=metavar, choices=o.choices)
        elif o.typespec == Sequence[str]:
            parser.add_argument(*flags, action='append', type=str, dest=optname, help=o.help + ' May be passed multiple times.', metavar=metavar, choices=o.choices)
        else:
            raise ValueError('Unsupported option type: %s', o.typespec)

def dump_defaults(opts, out: TextIO):
    if False:
        while True:
            i = 10
    '\n    Dumps an annotated file with all options.\n    '
    s = ruamel.yaml.comments.CommentedMap()
    for k in sorted(opts.keys()):
        o = opts._options[k]
        s[k] = o.default
        txt = o.help.strip()
        if o.choices:
            txt += ' Valid values are %s.' % ', '.join((repr(c) for c in o.choices))
        else:
            t = typecheck.typespec_to_str(o.typespec)
            txt += ' Type %s.' % t
        txt = '\n'.join(textwrap.wrap(txt))
        s.yaml_set_comment_before_after_key(k, before='\n' + txt)
    return ruamel.yaml.YAML().dump(s, out)

def dump_dicts(opts, keys: Iterable[str] | None=None) -> dict:
    if False:
        print('Hello World!')
    '\n    Dumps the options into a list of dict object.\n\n    Return: A list like: { "anticache": { type: "bool", default: false, value: true, help: "help text"} }\n    '
    options_dict = {}
    if keys is None:
        keys = opts.keys()
    for k in sorted(keys):
        o = opts._options[k]
        t = typecheck.typespec_to_str(o.typespec)
        option = {'type': t, 'default': o.default, 'value': o.current(), 'help': o.help, 'choices': o.choices}
        options_dict[k] = option
    return options_dict

def parse(text):
    if False:
        i = 10
        return i + 15
    if not text:
        return {}
    try:
        yaml = ruamel.yaml.YAML(typ='unsafe', pure=True)
        data = yaml.load(text)
    except ruamel.yaml.error.YAMLError as v:
        if hasattr(v, 'problem_mark'):
            snip = v.problem_mark.get_snippet()
            raise exceptions.OptionsError('Config error at line %s:\n%s\n%s' % (v.problem_mark.line + 1, snip, getattr(v, 'problem', '')))
        else:
            raise exceptions.OptionsError('Could not parse options.')
    if isinstance(data, str):
        raise exceptions.OptionsError('Config error - no keys found.')
    elif data is None:
        return {}
    return data

def load(opts: OptManager, text: str) -> None:
    if False:
        print('Hello World!')
    '\n    Load configuration from text, over-writing options already set in\n    this object. May raise OptionsError if the config file is invalid.\n    '
    data = parse(text)
    opts.update_defer(**data)

def load_paths(opts: OptManager, *paths: str) -> None:
    if False:
        while True:
            i = 10
    "\n    Load paths in order. Each path takes precedence over the previous\n    path. Paths that don't exist are ignored, errors raise an\n    OptionsError.\n    "
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.exists(p) and os.path.isfile(p):
            with open(p, encoding='utf8') as f:
                try:
                    txt = f.read()
                except UnicodeDecodeError as e:
                    raise exceptions.OptionsError(f'Error reading {p}: {e}')
            try:
                load(opts, txt)
            except exceptions.OptionsError as e:
                raise exceptions.OptionsError(f'Error reading {p}: {e}')

def serialize(opts: OptManager, file: TextIO, text: str, defaults: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Performs a round-trip serialization. If text is not None, it is\n    treated as a previous serialization that should be modified\n    in-place.\n\n    - If "defaults" is False, only options with non-default values are\n        serialized. Default values in text are preserved.\n    - Unknown options in text are removed.\n    - Raises OptionsError if text is invalid.\n    '
    data = parse(text)
    for k in opts.keys():
        if defaults or opts.has_changed(k):
            data[k] = getattr(opts, k)
    for k in list(data.keys()):
        if k not in opts._options:
            del data[k]
    ruamel.yaml.YAML().dump(data, file)

def save(opts: OptManager, path: str, defaults: bool=False) -> None:
    if False:
        return 10
    '\n    Save to path. If the destination file exists, modify it in-place.\n\n    Raises OptionsError if the existing data is corrupt.\n    '
    path = os.path.expanduser(path)
    if os.path.exists(path) and os.path.isfile(path):
        with open(path, encoding='utf8') as f:
            try:
                data = f.read()
            except UnicodeDecodeError as e:
                raise exceptions.OptionsError(f'Error trying to modify {path}: {e}')
    else:
        data = ''
    with open(path, 'w', encoding='utf8') as f:
        serialize(opts, f, data, defaults)