"""This module contains the _EstimatorPrettyPrinter class used in
BaseEstimator.__repr__ for pretty-printing estimators"""
import inspect
import pprint
from collections import OrderedDict
from .._config import get_config
from ..base import BaseEstimator
from . import is_scalar_nan

class KeyValTuple(tuple):
    """Dummy class for correctly rendering key-value tuples from dicts."""

    def __repr__(self):
        if False:
            while True:
                i = 10
        return super().__repr__()

class KeyValTupleParam(KeyValTuple):
    """Dummy class for correctly rendering key-value tuples from parameters."""
    pass

def _changed_params(estimator):
    if False:
        print('Hello World!')
    'Return dict (param_name: value) of parameters that were given to\n    estimator with non-default values.'
    params = estimator.get_params(deep=False)
    init_func = getattr(estimator.__init__, 'deprecated_original', estimator.__init__)
    init_params = inspect.signature(init_func).parameters
    init_params = {name: param.default for (name, param) in init_params.items()}

    def has_changed(k, v):
        if False:
            while True:
                i = 10
        if k not in init_params:
            return True
        if init_params[k] == inspect._empty:
            return True
        if isinstance(v, BaseEstimator) and v.__class__ != init_params[k].__class__:
            return True
        if repr(v) != repr(init_params[k]) and (not (is_scalar_nan(init_params[k]) and is_scalar_nan(v))):
            return True
        return False
    return {k: v for (k, v) in params.items() if has_changed(k, v)}

class _EstimatorPrettyPrinter(pprint.PrettyPrinter):
    """Pretty Printer class for estimator objects.

    This extends the pprint.PrettyPrinter class, because:
    - we need estimators to be printed with their parameters, e.g.
      Estimator(param1=value1, ...) which is not supported by default.
    - the 'compact' parameter of PrettyPrinter is ignored for dicts, which
      may lead to very long representations that we want to avoid.

    Quick overview of pprint.PrettyPrinter (see also
    https://stackoverflow.com/questions/49565047/pprint-with-hex-numbers):

    - the entry point is the _format() method which calls format() (overridden
      here)
    - format() directly calls _safe_repr() for a first try at rendering the
      object
    - _safe_repr formats the whole object recursively, only calling itself,
      not caring about line length or anything
    - back to _format(), if the output string is too long, _format() then calls
      the appropriate _pprint_TYPE() method (e.g. _pprint_list()) depending on
      the type of the object. This where the line length and the compact
      parameters are taken into account.
    - those _pprint_TYPE() methods will internally use the format() method for
      rendering the nested objects of an object (e.g. the elements of a list)

    In the end, everything has to be implemented twice: in _safe_repr and in
    the custom _pprint_TYPE methods. Unfortunately PrettyPrinter is really not
    straightforward to extend (especially when we want a compact output), so
    the code is a bit convoluted.

    This class overrides:
    - format() to support the changed_only parameter
    - _safe_repr to support printing of estimators (for when they fit on a
      single line)
    - _format_dict_items so that dict are correctly 'compacted'
    - _format_items so that ellipsis is used on long lists and tuples

    When estimators cannot be printed on a single line, the builtin _format()
    will call _pprint_estimator() because it was registered to do so (see
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator).

    both _format_dict_items() and _pprint_estimator() use the
    _format_params_or_dict_items() method that will format parameters and
    key-value pairs respecting the compact parameter. This method needs another
    subroutine _pprint_key_val_tuple() used when a parameter or a key-value
    pair is too long to fit on a single line. This subroutine is called in
    _format() and is registered as well in the _dispatch dict (just like
    _pprint_estimator). We had to create the two classes KeyValTuple and
    KeyValTupleParam for this.
    """

    def __init__(self, indent=1, width=80, depth=None, stream=None, *, compact=False, indent_at_name=True, n_max_elements_to_show=None):
        if False:
            return 10
        super().__init__(indent, width, depth, stream, compact=compact)
        self._indent_at_name = indent_at_name
        if self._indent_at_name:
            self._indent_per_level = 1
        self._changed_only = get_config()['print_changed_only']
        self.n_max_elements_to_show = n_max_elements_to_show

    def format(self, object, context, maxlevels, level):
        if False:
            return 10
        return _safe_repr(object, context, maxlevels, level, changed_only=self._changed_only)

    def _pprint_estimator(self, object, stream, indent, allowance, context, level):
        if False:
            for i in range(10):
                print('nop')
        stream.write(object.__class__.__name__ + '(')
        if self._indent_at_name:
            indent += len(object.__class__.__name__)
        if self._changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)
        params = OrderedDict(((name, val) for (name, val) in sorted(params.items())))
        self._format_params(params.items(), stream, indent, allowance + 1, context, level)
        stream.write(')')

    def _format_dict_items(self, items, stream, indent, allowance, context, level):
        if False:
            print('Hello World!')
        return self._format_params_or_dict_items(items, stream, indent, allowance, context, level, is_dict=True)

    def _format_params(self, items, stream, indent, allowance, context, level):
        if False:
            return 10
        return self._format_params_or_dict_items(items, stream, indent, allowance, context, level, is_dict=False)

    def _format_params_or_dict_items(self, object, stream, indent, allowance, context, level, is_dict):
        if False:
            while True:
                i = 10
        "Format dict items or parameters respecting the compact=True\n        parameter. For some reason, the builtin rendering of dict items doesn't\n        respect compact=True and will use one line per key-value if all cannot\n        fit in a single line.\n        Dict items will be rendered as <'key': value> while params will be\n        rendered as <key=value>. The implementation is mostly copy/pasting from\n        the builtin _format_items().\n        This also adds ellipsis if the number of items is greater than\n        self.n_max_elements_to_show.\n        "
        write = stream.write
        indent += self._indent_per_level
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(object)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(', ...')
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                (k, v) = ent
                krepr = self._repr(k, context, level)
                vrepr = self._repr(v, context, level)
                if not is_dict:
                    krepr = krepr.strip("'")
                middle = ': ' if is_dict else '='
                rep = krepr + middle + vrepr
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            class_ = KeyValTuple if is_dict else KeyValTupleParam
            self._format(class_(ent), stream, indent, allowance if last else 1, context, level)

    def _format_items(self, items, stream, indent, allowance, context, level):
        if False:
            return 10
        'Format the items of an iterable (list, tuple...). Same as the\n        built-in _format_items, with support for ellipsis if the number of\n        elements is greater than self.n_max_elements_to_show.\n        '
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(', ...')
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent, allowance if last else 1, context, level)

    def _pprint_key_val_tuple(self, object, stream, indent, allowance, context, level):
        if False:
            i = 10
            return i + 15
        'Pretty printing for key-value tuples from dict or parameters.'
        (k, v) = object
        rep = self._repr(k, context, level)
        if isinstance(object, KeyValTupleParam):
            rep = rep.strip("'")
            middle = '='
        else:
            middle = ': '
        stream.write(rep)
        stream.write(middle)
        self._format(v, stream, indent + len(rep) + len(middle), allowance, context, level)
    _dispatch = pprint.PrettyPrinter._dispatch.copy()
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator
    _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple

def _safe_repr(object, context, maxlevels, level, changed_only=False):
    if False:
        print('Hello World!')
    'Same as the builtin _safe_repr, with added support for Estimator\n    objects.'
    typ = type(object)
    if typ in pprint._builtin_scalars:
        return (repr(object), True, False)
    r = getattr(typ, '__repr__', None)
    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return ('{}', True, False)
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return ('{...}', False, objid in context)
        if objid in context:
            return (pprint._recursion(object), False, True)
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(object.items(), key=pprint._safe_tuple)
        for (k, v) in items:
            (krepr, kreadable, krecur) = saferepr(k, context, maxlevels, level, changed_only=changed_only)
            (vrepr, vreadable, vrecur) = saferepr(v, context, maxlevels, level, changed_only=changed_only)
            append('%s: %s' % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ('{%s}' % ', '.join(components), readable, recursive)
    if issubclass(typ, list) and r is list.__repr__ or (issubclass(typ, tuple) and r is tuple.__repr__):
        if issubclass(typ, list):
            if not object:
                return ('[]', True, False)
            format = '[%s]'
        elif len(object) == 1:
            format = '(%s,)'
        else:
            if not object:
                return ('()', True, False)
            format = '(%s)'
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return (format % '...', False, objid in context)
        if objid in context:
            return (pprint._recursion(object), False, True)
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            (orepr, oreadable, orecur) = _safe_repr(o, context, maxlevels, level, changed_only=changed_only)
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return (format % ', '.join(components), readable, recursive)
    if issubclass(typ, BaseEstimator):
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return ('{...}', False, objid in context)
        if objid in context:
            return (pprint._recursion(object), False, True)
        context[objid] = 1
        readable = True
        recursive = False
        if changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(params.items(), key=pprint._safe_tuple)
        for (k, v) in items:
            (krepr, kreadable, krecur) = saferepr(k, context, maxlevels, level, changed_only=changed_only)
            (vrepr, vreadable, vrecur) = saferepr(v, context, maxlevels, level, changed_only=changed_only)
            append('%s=%s' % (krepr.strip("'"), vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ('%s(%s)' % (typ.__name__, ', '.join(components)), readable, recursive)
    rep = repr(object)
    return (rep, rep and (not rep.startswith('<')), False)