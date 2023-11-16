"""Utility functions for writing decorators (which modify docstrings)."""
import sys

def get_qualified_name(function):
    if False:
        print('Hello World!')
    if hasattr(function, '__qualname__'):
        return function.__qualname__
    if hasattr(function, 'im_class'):
        return function.im_class.__name__ + '.' + function.__name__
    return function.__name__

def _normalize_docstring(docstring):
    if False:
        print('Hello World!')
    'Normalizes the docstring.\n\n  Replaces tabs with spaces, removes leading and trailing blanks lines, and\n  removes any indentation.\n\n  Copied from PEP-257:\n  https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation\n\n  Args:\n    docstring: the docstring to normalize\n\n  Returns:\n    The normalized docstring\n  '
    if not docstring:
        return ''
    lines = docstring.expandtabs().splitlines()
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    while trimmed and (not trimmed[-1]):
        trimmed.pop()
    while trimmed and (not trimmed[0]):
        trimmed.pop(0)
    return '\n'.join(trimmed)

def add_notice_to_docstring(doc, instructions, no_doc_str, suffix_str, notice, notice_type='Warning'):
    if False:
        while True:
            i = 10
    'Adds a deprecation notice to a docstring.\n\n  Args:\n    doc: The original docstring.\n    instructions: A string, describing how to fix the problem.\n    no_doc_str: The default value to use for `doc` if `doc` is empty.\n    suffix_str: Is added to the end of the first line.\n    notice: A list of strings. The main notice warning body.\n    notice_type: The type of notice to use. Should be one of `[Caution,\n    Deprecated, Important, Note, Warning]`\n\n  Returns:\n    A new docstring, with the notice attached.\n\n  Raises:\n    ValueError: If `notice` is empty.\n  '
    allowed_notice_types = ['Deprecated', 'Warning', 'Caution', 'Important', 'Note']
    if notice_type not in allowed_notice_types:
        raise ValueError(f'Unrecognized notice type. Should be one of: {allowed_notice_types}')
    if not doc:
        lines = [no_doc_str]
    else:
        lines = _normalize_docstring(doc).splitlines()
        lines[0] += ' ' + suffix_str
    if not notice:
        raise ValueError('The `notice` arg must not be empty.')
    notice[0] = f'{notice_type}: {notice[0]}'
    notice = [''] + notice + ([instructions] if instructions else [])
    if len(lines) > 1:
        if lines[1].strip():
            notice.append('')
        lines[1:1] = notice
    else:
        lines += notice
    return '\n'.join(lines)

def validate_callable(func, decorator_name):
    if False:
        print('Hello World!')
    if not hasattr(func, '__call__'):
        raise ValueError('%s is not a function. If this is a property, make sure @property appears before @%s in your source code:\n\n@property\n@%s\ndef method(...)' % (func, decorator_name, decorator_name))

class classproperty(object):
    """Class property decorator.

  Example usage:

  class MyClass(object):

    @classproperty
    def value(cls):
      return '123'

  > print MyClass.value
  123
  """

    def __init__(self, func):
        if False:
            i = 10
            return i + 15
        self._func = func

    def __get__(self, owner_self, owner_cls):
        if False:
            while True:
                i = 10
        return self._func(owner_cls)

class _CachedClassProperty(object):
    """Cached class property decorator.

  Transforms a class method into a property whose value is computed once
  and then cached as a normal attribute for the life of the class.  Example
  usage:

  >>> class MyClass(object):
  ...   @cached_classproperty
  ...   def value(cls):
  ...     print("Computing value")
  ...     return '<property of %s>' % cls.__name__
  >>> class MySubclass(MyClass):
  ...   pass
  >>> MyClass.value
  Computing value
  '<property of MyClass>'
  >>> MyClass.value  # uses cached value
  '<property of MyClass>'
  >>> MySubclass.value
  Computing value
  '<property of MySubclass>'

  This decorator is similar to `functools.cached_property`, but it adds a
  property to the class, not to individual instances.
  """

    def __init__(self, func):
        if False:
            return 10
        self._func = func
        self._cache = {}

    def __get__(self, obj, objtype):
        if False:
            for i in range(10):
                print('nop')
        if objtype not in self._cache:
            self._cache[objtype] = self._func(objtype)
        return self._cache[objtype]

    def __set__(self, obj, value):
        if False:
            return 10
        raise AttributeError('property %s is read-only' % self._func.__name__)

    def __delete__(self, obj):
        if False:
            return 10
        raise AttributeError('property %s is read-only' % self._func.__name__)

def cached_classproperty(func):
    if False:
        print('Hello World!')
    return _CachedClassProperty(func)
cached_classproperty.__doc__ = _CachedClassProperty.__doc__