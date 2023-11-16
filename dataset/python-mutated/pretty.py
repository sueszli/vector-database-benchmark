"""
Python advanced pretty printer.  This pretty printer is intended to
replace the old `pprint` python module which does not allow developers
to provide their own pretty print callbacks.
This module is based on ruby's `prettyprint.rb` library by `Tanaka Akira`.
Example Usage
-------------
To get a string of the output use `pretty`::
    from pretty import pretty
    string = pretty(complex_object)
Extending
---------
The pretty library allows developers to add pretty printing rules for their
own objects.  This process is straightforward.  All you have to do is to
add a `_repr_pretty_` method to your object and call the methods on the
pretty printer passed::
    class MyObject(object):
        def _repr_pretty_(self, p, cycle):
            ...
Here is an example implementation of a `_repr_pretty_` method for a list
subclass::
    class MyList(list):
        def _repr_pretty_(self, p, cycle):
            if cycle:
                p.text('MyList(...)')
            else:
                with p.group(8, 'MyList([', '])'):
                    for idx, item in enumerate(self):
                        if idx:
                            p.text(',')
                            p.breakable()
                        p.pretty(item)
The `cycle` parameter is `True` if pretty detected a cycle.  You *have* to
react to that or the result is an infinite loop.  `p.text()` just adds
non breaking text to the output, `p.breakable()` either adds a whitespace
or breaks here.  If you pass it an argument it's used instead of the
default space.  `p.pretty` prettyprints another object using the pretty print
method.
The first parameter to the `group` function specifies the extra indentation
of the next line.  In this example the next item will either be on the same
line (if the items are short enough) or aligned with the right edge of the
opening bracket of `MyList`.
If you just want to indent something you can use the group function
without open / close parameters.  You can also use this code::
    with p.indent(2):
        ...
Inheritance diagram:
.. inheritance-diagram:: IPython.lib.pretty
   :parts: 3
:copyright: 2007 by Armin Ronacher.
            Portions (c) 2009 by Robert Kern.
:license: BSD License.
"""
import datetime
import re
import struct
import sys
import types
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Flag
from io import StringIO
from math import copysign, isnan
__all__ = ['pretty', 'IDKey', 'RepresentationPrinter']

def _safe_getattr(obj, attr, default=None):
    if False:
        for i in range(10):
            print('nop')
    'Safe version of getattr.\n\n    Same as getattr, but will return ``default`` on any Exception,\n    rather than raising.\n\n    '
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

def pretty(obj):
    if False:
        i = 10
        return i + 15
    "Pretty print the object's representation."
    printer = RepresentationPrinter()
    printer.pretty(obj)
    return printer.getvalue()

class IDKey:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash((type(self), id(self.value)))

    def __eq__(self, __o: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(__o, type(self)) and id(self.value) == id(__o.value)

class RepresentationPrinter:
    """Special pretty printer that has a `pretty` method that calls the pretty
    printer for a python object.

    This class stores processing data on `self` so you must *never* use
    this class in a threaded environment.  Always lock it or
    reinstantiate it.

    """

    def __init__(self, output=None, *, context=None):
        if False:
            return 10
        'Pass the output stream, and optionally the current build context.\n\n        We use the context to represent objects constructed by strategies by showing\n        *how* they were constructed, and add annotations showing which parts of the\n        minimal failing example can vary without changing the test result.\n        '
        self.broken = False
        self.output = StringIO() if output is None else output
        self.max_width = 79
        self.max_seq_length = 1000
        self.output_width = 0
        self.buffer_width = 0
        self.buffer = deque()
        root_group = Group(0)
        self.group_stack = [root_group]
        self.group_queue = GroupQueue(root_group)
        self.indentation = 0
        self.snans = 0
        self.stack = []
        self.singleton_pprinters = {}
        self.type_pprinters = {}
        self.deferred_pprinters = {}
        if 'IPython.lib.pretty' in sys.modules:
            ipp = sys.modules['IPython.lib.pretty']
            self.singleton_pprinters.update(ipp._singleton_pprinters)
            self.type_pprinters.update(ipp._type_pprinters)
            self.deferred_pprinters.update(ipp._deferred_type_pprinters)
        self.singleton_pprinters.update(_singleton_pprinters)
        self.type_pprinters.update(_type_pprinters)
        self.deferred_pprinters.update(_deferred_type_pprinters)
        if context is None:
            self.known_object_printers = defaultdict(list)
            self.slice_comments = {}
        else:
            self.known_object_printers = context.known_object_printers
            self.slice_comments = context.data.slice_comments
        assert all((isinstance(k, IDKey) for k in self.known_object_printers))

    def pretty(self, obj):
        if False:
            while True:
                i = 10
        'Pretty print the given object.'
        obj_id = id(obj)
        cycle = obj_id in self.stack
        self.stack.append(obj_id)
        try:
            with self.group():
                obj_class = _safe_getattr(obj, '__class__', None) or type(obj)
                try:
                    printer = self.singleton_pprinters[obj_id]
                except (TypeError, KeyError):
                    pass
                else:
                    return printer(obj, self, cycle)
                for cls in obj_class.__mro__:
                    if cls in self.type_pprinters:
                        return self.type_pprinters[cls](obj, self, cycle)
                    else:
                        key = (_safe_getattr(cls, '__module__', None), _safe_getattr(cls, '__name__', None))
                        if key in self.deferred_pprinters:
                            printer = self.deferred_pprinters.pop(key)
                            self.type_pprinters[cls] = printer
                            return printer(obj, self, cycle)
                        elif '_repr_pretty_' in cls.__dict__:
                            meth = cls._repr_pretty_
                            if callable(meth):
                                return meth(obj, self, cycle)
                printers = self.known_object_printers[IDKey(obj)]
                if len(printers) == 1:
                    return printers[0](obj, self, cycle)
                elif printers:
                    strs = set()
                    for f in printers:
                        p = RepresentationPrinter()
                        f(obj, p, cycle)
                        strs.add(p.getvalue())
                    if len(strs) == 1:
                        return printers[0](obj, self, cycle)
                return _repr_pprint(obj, self, cycle)
        finally:
            self.stack.pop()

    def _break_outer_groups(self):
        if False:
            i = 10
            return i + 15
        while self.max_width < self.output_width + self.buffer_width:
            group = self.group_queue.deq()
            if not group:
                return
            while group.breakables:
                x = self.buffer.popleft()
                self.output_width = x.output(self.output, self.output_width)
                self.buffer_width -= x.width
            while self.buffer and isinstance(self.buffer[0], Text):
                x = self.buffer.popleft()
                self.output_width = x.output(self.output, self.output_width)
                self.buffer_width -= x.width

    def text(self, obj):
        if False:
            print('Hello World!')
        'Add literal text to the output.'
        width = len(obj)
        if self.buffer:
            text = self.buffer[-1]
            if not isinstance(text, Text):
                text = Text()
                self.buffer.append(text)
            text.add(obj, width)
            self.buffer_width += width
            self._break_outer_groups()
        else:
            self.output.write(obj)
            self.output_width += width

    def breakable(self, sep=' '):
        if False:
            while True:
                i = 10
        'Add a breakable separator to the output.\n\n        This does not mean that it will automatically break here.  If no\n        breaking on this position takes place the `sep` is inserted\n        which default to one space.\n\n        '
        width = len(sep)
        group = self.group_stack[-1]
        if group.want_break:
            self.flush()
            self.output.write('\n' + ' ' * self.indentation)
            self.output_width = self.indentation
            self.buffer_width = 0
        else:
            self.buffer.append(Breakable(sep, width, self))
            self.buffer_width += width
            self._break_outer_groups()

    def break_(self):
        if False:
            for i in range(10):
                print('nop')
        'Explicitly insert a newline into the output, maintaining correct\n        indentation.'
        self.flush()
        self.output.write('\n' + ' ' * self.indentation)
        self.output_width = self.indentation
        self.buffer_width = 0

    @contextmanager
    def indent(self, indent):
        if False:
            while True:
                i = 10
        '`with`-statement support for indenting/dedenting.'
        self.indentation += indent
        try:
            yield
        finally:
            self.indentation -= indent

    @contextmanager
    def group(self, indent=0, open='', close=''):
        if False:
            for i in range(10):
                print('nop')
        "Context manager for an indented group.\n\n            with p.group(1, '{', '}'):\n\n        The first parameter specifies the indentation for the next line\n        (usually the width of the opening text), the second and third the\n        opening and closing delimiters.\n        "
        self.begin_group(indent=indent, open=open)
        try:
            yield
        finally:
            self.end_group(dedent=indent, close=close)

    def begin_group(self, indent=0, open=''):
        if False:
            while True:
                i = 10
        'Use the `with group(...) context manager instead.\n\n        The begin_group() and end_group() methods are for IPython compatibility only;\n        see https://github.com/HypothesisWorks/hypothesis/issues/3721 for details.\n        '
        if open:
            self.text(open)
        group = Group(self.group_stack[-1].depth + 1)
        self.group_stack.append(group)
        self.group_queue.enq(group)
        self.indentation += indent

    def end_group(self, dedent=0, close=''):
        if False:
            return 10
        'See begin_group().'
        self.indentation -= dedent
        group = self.group_stack.pop()
        if not group.breakables:
            self.group_queue.remove(group)
        if close:
            self.text(close)

    def _enumerate(self, seq):
        if False:
            while True:
                i = 10
        'Like enumerate, but with an upper limit on the number of items.'
        for (idx, x) in enumerate(seq):
            if self.max_seq_length and idx >= self.max_seq_length:
                self.text(',')
                self.breakable()
                self.text('...')
                return
            yield (idx, x)

    def flush(self):
        if False:
            print('Hello World!')
        'Flush data that is left in the buffer.'
        if self.snans:
            snans = self.snans
            self.snans = 0
            self.breakable('  ')
            self.text(f'# Saw {snans} signaling NaN' + 's' * (snans > 1))
        for data in self.buffer:
            self.output_width += data.output(self.output, self.output_width)
        self.buffer.clear()
        self.buffer_width = 0

    def getvalue(self):
        if False:
            while True:
                i = 10
        assert isinstance(self.output, StringIO)
        self.flush()
        return self.output.getvalue()

    def repr_call(self, func_name, args, kwargs, *, force_split=None, arg_slices=None, leading_comment=None):
        if False:
            print('Hello World!')
        "Helper function to represent a function call.\n\n        - func_name, args, and kwargs should all be pretty obvious.\n        - If split_lines, we'll force one-argument-per-line; otherwise we'll place\n          calls that fit on a single line (and split otherwise).\n        - arg_slices is a mapping from pos-idx or keyword to (start_idx, end_idx)\n          of the Conjecture buffer, by which we can look up comments to add.\n        "
        assert isinstance(func_name, str)
        if func_name.startswith(('lambda:', 'lambda ')):
            func_name = f'({func_name})'
        self.text(func_name)
        all_args = [(None, v) for v in args] + list(kwargs.items())
        comments = {k: self.slice_comments[v] for (k, v) in (arg_slices or {}).items() if v in self.slice_comments}
        if leading_comment or any((k in comments for (k, _) in all_args)):
            force_split = True
        if force_split is None:
            p = RepresentationPrinter()
            p.stack = self.stack.copy()
            p.known_object_printers = self.known_object_printers
            p.repr_call('_' * self.output_width, args, kwargs, force_split=False)
            s = p.getvalue()
            force_split = '\n' in s
        with self.group(indent=4, open='(', close=''):
            for (i, (k, v)) in enumerate(all_args):
                if force_split:
                    if i == 0 and leading_comment:
                        self.break_()
                        self.text(leading_comment)
                    self.break_()
                else:
                    self.breakable(' ' if i else '')
                if k:
                    self.text(f'{k}=')
                self.pretty(v)
                if force_split or i + 1 < len(all_args):
                    self.text(',')
                comment = comments.get(i) or comments.get(k)
                if comment:
                    self.text(f'  # {comment}')
        if all_args and force_split:
            self.break_()
        self.text(')')

class Printable:

    def output(self, stream, output_width):
        if False:
            return 10
        raise NotImplementedError

class Text(Printable):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.objs = []
        self.width = 0

    def output(self, stream, output_width):
        if False:
            while True:
                i = 10
        for obj in self.objs:
            stream.write(obj)
        return output_width + self.width

    def add(self, obj, width):
        if False:
            for i in range(10):
                print('nop')
        self.objs.append(obj)
        self.width += width

class Breakable(Printable):

    def __init__(self, seq, width, pretty):
        if False:
            for i in range(10):
                print('nop')
        self.obj = seq
        self.width = width
        self.pretty = pretty
        self.indentation = pretty.indentation
        self.group = pretty.group_stack[-1]
        self.group.breakables.append(self)

    def output(self, stream, output_width):
        if False:
            for i in range(10):
                print('nop')
        self.group.breakables.popleft()
        if self.group.want_break:
            stream.write('\n' + ' ' * self.indentation)
            return self.indentation
        if not self.group.breakables:
            self.pretty.group_queue.remove(self.group)
        stream.write(self.obj)
        return output_width + self.width

class Group(Printable):

    def __init__(self, depth):
        if False:
            i = 10
            return i + 15
        self.depth = depth
        self.breakables = deque()
        self.want_break = False

class GroupQueue:

    def __init__(self, *groups):
        if False:
            while True:
                i = 10
        self.queue = []
        for group in groups:
            self.enq(group)

    def enq(self, group):
        if False:
            while True:
                i = 10
        depth = group.depth
        while depth > len(self.queue) - 1:
            self.queue.append([])
        self.queue[depth].append(group)

    def deq(self):
        if False:
            i = 10
            return i + 15
        for stack in self.queue:
            for (idx, group) in enumerate(reversed(stack)):
                if group.breakables:
                    del stack[idx]
                    group.want_break = True
                    return group
            for group in stack:
                group.want_break = True
            del stack[:]

    def remove(self, group):
        if False:
            return 10
        try:
            self.queue[group.depth].remove(group)
        except ValueError:
            pass

def _seq_pprinter_factory(start, end, basetype):
    if False:
        while True:
            i = 10
    'Factory that returns a pprint function useful for sequences.\n\n    Used by the default pprint for tuples, dicts, and lists.\n    '

    def inner(obj, p, cycle):
        if False:
            while True:
                i = 10
        typ = type(obj)
        if basetype is not None and typ is not basetype and (typ.__repr__ != basetype.__repr__):
            return p.text(typ.__repr__(obj))
        if cycle:
            return p.text(start + '...' + end)
        step = len(start)
        with p.group(step, start, end):
            for (idx, x) in p._enumerate(obj):
                if idx:
                    p.text(',')
                    p.breakable()
                p.pretty(x)
            if len(obj) == 1 and type(obj) is tuple:
                p.text(',')
    return inner

def _set_pprinter_factory(start, end, basetype):
    if False:
        for i in range(10):
            print('nop')
    'Factory that returns a pprint function useful for sets and\n    frozensets.'

    def inner(obj, p, cycle):
        if False:
            while True:
                i = 10
        typ = type(obj)
        if basetype is not None and typ is not basetype and (typ.__repr__ != basetype.__repr__):
            return p.text(typ.__repr__(obj))
        if cycle:
            return p.text(start + '...' + end)
        if not obj:
            p.text(basetype.__name__ + '()')
        else:
            step = len(start)
            with p.group(step, start, end):
                items = obj
                if not (p.max_seq_length and len(obj) >= p.max_seq_length):
                    try:
                        items = sorted(obj)
                    except Exception:
                        pass
                for (idx, x) in p._enumerate(items):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(x)
    return inner

def _dict_pprinter_factory(start, end, basetype=None):
    if False:
        return 10
    'Factory that returns a pprint function used by the default pprint of\n    dicts and dict proxies.'

    def inner(obj, p, cycle):
        if False:
            while True:
                i = 10
        typ = type(obj)
        if basetype is not None and typ is not basetype and (typ.__repr__ != basetype.__repr__):
            return p.text(typ.__repr__(obj))
        if cycle:
            return p.text('{...}')
        with p.group(1, start, end):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', BytesWarning)
                for (idx, key) in p._enumerate(obj):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(key)
                    p.text(': ')
                    p.pretty(obj[key])
    inner.__name__ = f'_dict_pprinter_factory({start!r}, {end!r}, {basetype!r})'
    return inner

def _super_pprint(obj, p, cycle):
    if False:
        for i in range(10):
            print('nop')
    'The pprint for the super type.'
    with p.group(8, '<super: ', '>'):
        p.pretty(obj.__thisclass__)
        p.text(',')
        p.breakable()
        p.pretty(obj.__self__)

def _re_pattern_pprint(obj, p, cycle):
    if False:
        print('Hello World!')
    'The pprint function for regular expression patterns.'
    p.text('re.compile(')
    pattern = repr(obj.pattern)
    if pattern[:1] in 'uU':
        pattern = pattern[1:]
        prefix = 'ur'
    else:
        prefix = 'r'
    pattern = prefix + pattern.replace('\\\\', '\\')
    p.text(pattern)
    if obj.flags:
        p.text(',')
        p.breakable()
        done_one = False
        for flag in ('TEMPLATE', 'IGNORECASE', 'LOCALE', 'MULTILINE', 'DOTALL', 'UNICODE', 'VERBOSE', 'DEBUG'):
            if obj.flags & getattr(re, flag, 0):
                if done_one:
                    p.text('|')
                p.text('re.' + flag)
                done_one = True
    p.text(')')

def _type_pprint(obj, p, cycle):
    if False:
        for i in range(10):
            print('nop')
    'The pprint for classes and types.'
    if type(obj).__repr__ != type.__repr__:
        _repr_pprint(obj, p, cycle)
        return
    mod = _safe_getattr(obj, '__module__', None)
    try:
        name = obj.__qualname__
        if not isinstance(name, str):
            raise Exception('Try __name__')
    except Exception:
        name = obj.__name__
        if not isinstance(name, str):
            name = '<unknown type>'
    if mod in (None, '__builtin__', 'builtins', 'exceptions'):
        p.text(name)
    else:
        p.text(mod + '.' + name)

def _repr_pprint(obj, p, cycle):
    if False:
        while True:
            i = 10
    'A pprint that just redirects to the normal repr function.'
    output = repr(obj)
    for (idx, output_line) in enumerate(output.splitlines()):
        if idx:
            p.break_()
        p.text(output_line)

def _function_pprint(obj, p, cycle):
    if False:
        while True:
            i = 10
    'Base pprint for all functions and builtin functions.'
    from hypothesis.internal.reflection import get_pretty_function_description
    p.text(get_pretty_function_description(obj))

def _exception_pprint(obj, p, cycle):
    if False:
        for i in range(10):
            print('nop')
    'Base pprint for all exceptions.'
    name = getattr(obj.__class__, '__qualname__', obj.__class__.__name__)
    if obj.__class__.__module__ not in ('exceptions', 'builtins'):
        name = f'{obj.__class__.__module__}.{name}'
    step = len(name) + 1
    with p.group(step, name + '(', ')'):
        for (idx, arg) in enumerate(getattr(obj, 'args', ())):
            if idx:
                p.text(',')
                p.breakable()
            p.pretty(arg)

def _repr_float_counting_nans(obj, p, cycle):
    if False:
        i = 10
        return i + 15
    if isnan(obj) and hasattr(p, 'snans'):
        if struct.pack('!d', abs(obj)) != struct.pack('!d', float('nan')):
            p.snans += 1
        if copysign(1.0, obj) == -1.0:
            p.text('-nan')
            return
    p.text(repr(obj))
_type_pprinters = {int: _repr_pprint, float: _repr_float_counting_nans, str: _repr_pprint, tuple: _seq_pprinter_factory('(', ')', tuple), list: _seq_pprinter_factory('[', ']', list), dict: _dict_pprinter_factory('{', '}', dict), set: _set_pprinter_factory('{', '}', set), frozenset: _set_pprinter_factory('frozenset({', '})', frozenset), super: _super_pprint, re.Pattern: _re_pattern_pprint, type: _type_pprint, types.FunctionType: _function_pprint, types.BuiltinFunctionType: _function_pprint, types.MethodType: _repr_pprint, datetime.datetime: _repr_pprint, datetime.timedelta: _repr_pprint, BaseException: _exception_pprint, slice: _repr_pprint, range: _repr_pprint, bytes: _repr_pprint}
_deferred_type_pprinters = {}

def for_type_by_name(type_module, type_name, func):
    if False:
        i = 10
        return i + 15
    'Add a pretty printer for a type specified by the module and name of a\n    type rather than the type object itself.'
    key = (type_module, type_name)
    oldfunc = _deferred_type_pprinters.get(key, None)
    _deferred_type_pprinters[key] = func
    return oldfunc
_singleton_pprinters = dict.fromkeys(map(id, [None, True, False, Ellipsis, NotImplemented]), _repr_pprint)

def _defaultdict_pprint(obj, p, cycle):
    if False:
        return 10
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + '(', ')'):
        if cycle:
            p.text('...')
        else:
            p.pretty(obj.default_factory)
            p.text(',')
            p.breakable()
            p.pretty(dict(obj))

def _ordereddict_pprint(obj, p, cycle):
    if False:
        i = 10
        return i + 15
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + '(', ')'):
        if cycle:
            p.text('...')
        elif obj:
            p.pretty(list(obj.items()))

def _deque_pprint(obj, p, cycle):
    if False:
        i = 10
        return i + 15
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + '(', ')'):
        if cycle:
            p.text('...')
        else:
            p.pretty(list(obj))

def _counter_pprint(obj, p, cycle):
    if False:
        i = 10
        return i + 15
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + '(', ')'):
        if cycle:
            p.text('...')
        elif obj:
            p.pretty(dict(obj))

def _repr_dataframe(obj, p, cycle):
    if False:
        while True:
            i = 10
    with p.indent(4):
        p.break_()
        _repr_pprint(obj, p, cycle)
    p.break_()

def _repr_enum(obj, p, cycle):
    if False:
        return 10
    tname = type(obj).__name__
    if isinstance(obj, Flag):
        p.text(' | '.join((f'{tname}.{x.name}' for x in type(obj) if x & obj == x)) or f'{tname}({obj.value!r})')
    else:
        p.text(f'{tname}.{obj.name}')
for_type_by_name('collections', 'defaultdict', _defaultdict_pprint)
for_type_by_name('collections', 'OrderedDict', _ordereddict_pprint)
for_type_by_name('ordereddict', 'OrderedDict', _ordereddict_pprint)
for_type_by_name('collections', 'deque', _deque_pprint)
for_type_by_name('collections', 'Counter', _counter_pprint)
for_type_by_name('pandas.core.frame', 'DataFrame', _repr_dataframe)
for_type_by_name('enum', 'Enum', _repr_enum)