"""The ``namedutils`` module defines two lightweight container types:
:class:`namedtuple` and :class:`namedlist`. Both are subtypes of built-in
sequence types, which are very fast and efficient. They simply add
named attribute accessors for specific indexes within themselves.

The :class:`namedtuple` is identical to the built-in
:class:`collections.namedtuple`, with a couple of enhancements,
including a ``__repr__`` more suitable to inheritance.

The :class:`namedlist` is the mutable counterpart to the
:class:`namedtuple`, and is much faster and lighter-weight than
full-blown :class:`object`. Consider this if you're implementing nodes
in a tree, graph, or other mutable data structure. If you want an even
skinnier approach, you'll probably have to look to C.
"""
from __future__ import print_function
import sys as _sys
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
from keyword import iskeyword as _iskeyword
from operator import itemgetter as _itemgetter
try:
    basestring

    def exec_(code, global_env):
        if False:
            while True:
                i = 10
        exec('exec code in global_env')
except NameError:
    basestring = (str, bytes)

    def exec_(code, global_env):
        if False:
            return 10
        exec(code, global_env)
__all__ = ['namedlist', 'namedtuple']
_repr_tmpl = '{name}=%r'
_imm_field_tmpl = "    {name} = _property(_itemgetter({index:d}), doc='Alias for field {index:d}')\n"
_m_field_tmpl = "    {name} = _property(_itemgetter({index:d}), _itemsetter({index:d}), doc='Alias for field {index:d}')\n"
_namedtuple_tmpl = "class {typename}(tuple):\n    '{typename}({arg_list})'\n\n    __slots__ = ()\n\n    _fields = {field_names!r}\n\n    def __new__(_cls, {arg_list}):  # TODO: tweak sig to make more extensible\n        'Create new instance of {typename}({arg_list})'\n        return _tuple.__new__(_cls, ({arg_list}))\n\n    @classmethod\n    def _make(cls, iterable, new=_tuple.__new__, len=len):\n        'Make a new {typename} object from a sequence or iterable'\n        result = new(cls, iterable)\n        if len(result) != {num_fields:d}:\n            raise TypeError('Expected {num_fields:d}'\n                            ' arguments, got %d' % len(result))\n        return result\n\n    def __repr__(self):\n        'Return a nicely formatted representation string'\n        tmpl = self.__class__.__name__ + '({repr_fmt})'\n        return tmpl % self\n\n    def _asdict(self):\n        'Return a new OrderedDict which maps field names to their values'\n        return OrderedDict(zip(self._fields, self))\n\n    def _replace(_self, **kwds):\n        'Return a new {typename} object replacing field(s) with new values'\n        result = _self._make(map(kwds.pop, {field_names!r}, _self))\n        if kwds:\n            raise ValueError('Got unexpected field names: %r' % kwds.keys())\n        return result\n\n    def __getnewargs__(self):\n        'Return self as a plain tuple.  Used by copy and pickle.'\n        return tuple(self)\n\n    __dict__ = _property(_asdict)\n\n    def __getstate__(self):\n        'Exclude the OrderedDict from pickling'  # wat\n        pass\n\n{field_defs}\n"

def namedtuple(typename, field_names, verbose=False, rename=False):
    if False:
        return 10
    "Returns a new subclass of tuple with named fields.\n\n    >>> Point = namedtuple('Point', ['x', 'y'])\n    >>> Point.__doc__                   # docstring for the new class\n    'Point(x, y)'\n    >>> p = Point(11, y=22)             # instantiate with pos args or keywords\n    >>> p[0] + p[1]                     # indexable like a plain tuple\n    33\n    >>> x, y = p                        # unpack like a regular tuple\n    >>> x, y\n    (11, 22)\n    >>> p.x + p.y                       # fields also accessible by name\n    33\n    >>> d = p._asdict()                 # convert to a dictionary\n    >>> d['x']\n    11\n    >>> Point(**d)                      # convert from a dictionary\n    Point(x=11, y=22)\n    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields\n    Point(x=100, y=22)\n    "
    if isinstance(field_names, basestring):
        field_names = field_names.replace(',', ' ').split()
    field_names = [str(x) for x in field_names]
    if rename:
        seen = set()
        for (index, name) in enumerate(field_names):
            if not all((c.isalnum() or c == '_' for c in name)) or _iskeyword(name) or (not name) or name[0].isdigit() or name.startswith('_') or (name in seen):
                field_names[index] = '_%d' % index
            seen.add(name)
    for name in [typename] + field_names:
        if not all((c.isalnum() or c == '_' for c in name)):
            raise ValueError('Type names and field names can only contain alphanumeric characters and underscores: %r' % name)
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a keyword: %r' % name)
        if name[0].isdigit():
            raise ValueError('Type names and field names cannot start with a number: %r' % name)
    seen = set()
    for name in field_names:
        if name.startswith('_') and (not rename):
            raise ValueError('Field names cannot start with an underscore: %r' % name)
        if name in seen:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen.add(name)
    fmt_kw = {'typename': typename}
    fmt_kw['field_names'] = tuple(field_names)
    fmt_kw['num_fields'] = len(field_names)
    fmt_kw['arg_list'] = repr(tuple(field_names)).replace("'", '')[1:-1]
    fmt_kw['repr_fmt'] = ', '.join((_repr_tmpl.format(name=name) for name in field_names))
    fmt_kw['field_defs'] = '\n'.join((_imm_field_tmpl.format(index=index, name=name) for (index, name) in enumerate(field_names)))
    class_definition = _namedtuple_tmpl.format(**fmt_kw)
    if verbose:
        print(class_definition)
    namespace = dict(_itemgetter=_itemgetter, __name__='namedtuple_%s' % typename, OrderedDict=OrderedDict, _property=property, _tuple=tuple)
    try:
        exec_(class_definition, namespace)
    except SyntaxError as e:
        raise SyntaxError(e.message + ':\n' + class_definition)
    result = namespace[typename]
    try:
        frame = _sys._getframe(1)
        result.__module__ = frame.f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    return result
_namedlist_tmpl = "class {typename}(list):\n    '{typename}({arg_list})'\n\n    __slots__ = ()\n\n    _fields = {field_names!r}\n\n    def __new__(_cls, {arg_list}):  # TODO: tweak sig to make more extensible\n        'Create new instance of {typename}({arg_list})'\n        return _list.__new__(_cls, ({arg_list}))\n\n    def __init__(self, {arg_list}):  # tuple didn't need this but list does\n        return _list.__init__(self, ({arg_list}))\n\n    @classmethod\n    def _make(cls, iterable, new=_list, len=len):\n        'Make a new {typename} object from a sequence or iterable'\n        # why did this function exist? why not just star the\n        # iterable like below?\n        result = cls(*iterable)\n        if len(result) != {num_fields:d}:\n            raise TypeError('Expected {num_fields:d} arguments,'\n                            ' got %d' % len(result))\n        return result\n\n    def __repr__(self):\n        'Return a nicely formatted representation string'\n        tmpl = self.__class__.__name__ + '({repr_fmt})'\n        return tmpl % tuple(self)\n\n    def _asdict(self):\n        'Return a new OrderedDict which maps field names to their values'\n        return OrderedDict(zip(self._fields, self))\n\n    def _replace(_self, **kwds):\n        'Return a new {typename} object replacing field(s) with new values'\n        result = _self._make(map(kwds.pop, {field_names!r}, _self))\n        if kwds:\n            raise ValueError('Got unexpected field names: %r' % kwds.keys())\n        return result\n\n    def __getnewargs__(self):\n        'Return self as a plain list.  Used by copy and pickle.'\n        return tuple(self)\n\n    __dict__ = _property(_asdict)\n\n    def __getstate__(self):\n        'Exclude the OrderedDict from pickling'  # wat\n        pass\n\n{field_defs}\n"

def namedlist(typename, field_names, verbose=False, rename=False):
    if False:
        while True:
            i = 10
    "Returns a new subclass of list with named fields.\n\n    >>> Point = namedlist('Point', ['x', 'y'])\n    >>> Point.__doc__                   # docstring for the new class\n    'Point(x, y)'\n    >>> p = Point(11, y=22)             # instantiate with pos args or keywords\n    >>> p[0] + p[1]                     # indexable like a plain list\n    33\n    >>> x, y = p                        # unpack like a regular list\n    >>> x, y\n    (11, 22)\n    >>> p.x + p.y                       # fields also accessible by name\n    33\n    >>> d = p._asdict()                 # convert to a dictionary\n    >>> d['x']\n    11\n    >>> Point(**d)                      # convert from a dictionary\n    Point(x=11, y=22)\n    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields\n    Point(x=100, y=22)\n    "
    if isinstance(field_names, basestring):
        field_names = field_names.replace(',', ' ').split()
    field_names = [str(x) for x in field_names]
    if rename:
        seen = set()
        for (index, name) in enumerate(field_names):
            if not all((c.isalnum() or c == '_' for c in name)) or _iskeyword(name) or (not name) or name[0].isdigit() or name.startswith('_') or (name in seen):
                field_names[index] = '_%d' % index
            seen.add(name)
    for name in [typename] + field_names:
        if not all((c.isalnum() or c == '_' for c in name)):
            raise ValueError('Type names and field names can only contain alphanumeric characters and underscores: %r' % name)
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a keyword: %r' % name)
        if name[0].isdigit():
            raise ValueError('Type names and field names cannot start with a number: %r' % name)
    seen = set()
    for name in field_names:
        if name.startswith('_') and (not rename):
            raise ValueError('Field names cannot start with an underscore: %r' % name)
        if name in seen:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen.add(name)
    fmt_kw = {'typename': typename}
    fmt_kw['field_names'] = tuple(field_names)
    fmt_kw['num_fields'] = len(field_names)
    fmt_kw['arg_list'] = repr(tuple(field_names)).replace("'", '')[1:-1]
    fmt_kw['repr_fmt'] = ', '.join((_repr_tmpl.format(name=name) for name in field_names))
    fmt_kw['field_defs'] = '\n'.join((_m_field_tmpl.format(index=index, name=name) for (index, name) in enumerate(field_names)))
    class_definition = _namedlist_tmpl.format(**fmt_kw)
    if verbose:
        print(class_definition)

    def _itemsetter(key):
        if False:
            for i in range(10):
                print('nop')

        def _itemsetter(obj, value):
            if False:
                print('Hello World!')
            obj[key] = value
        return _itemsetter
    namespace = dict(_itemgetter=_itemgetter, _itemsetter=_itemsetter, __name__='namedlist_%s' % typename, OrderedDict=OrderedDict, _property=property, _list=list)
    try:
        exec_(class_definition, namespace)
    except SyntaxError as e:
        raise SyntaxError(e.message + ':\n' + class_definition)
    result = namespace[typename]
    try:
        frame = _sys._getframe(1)
        result.__module__ = frame.f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    return result