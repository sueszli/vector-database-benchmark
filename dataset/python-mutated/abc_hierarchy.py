"""Hierarchy of abstract base classes, from _collections_abc.py."""
from pytype import utils
SUPERCLASSES = {'Hashable': [], 'Iterable': [], 'AsyncIterable': [], 'Sized': [], 'Callable': [], 'Awaitable': [], 'Iterator': ['Iterable'], 'AsyncIterator': ['AsyncIterable'], 'Coroutine': ['Awaitable'], 'Container': ['object'], 'Number': ['object'], 'Complex': ['Number'], 'Real': ['Complex'], 'Rational': ['Real'], 'Integral': ['Rational'], 'Set': ['Sized', 'Iterable', 'Container'], 'MutableSet': ['Set'], 'Mapping': ['Sized', 'Iterable', 'Container'], 'MappingView': ['Sized'], 'KeysView': ['MappingView', 'Set'], 'ItemsView': ['MappingView', 'Set'], 'ValuesView': ['MappingView'], 'MutableMapping': ['Mapping'], 'Sequence': ['Sized', 'Iterable', 'Container'], 'MutableSequence': ['Sequence'], 'ByteString': ['Sequence'], 'set': ['MutableSet'], 'frozenset': ['Set'], 'dict': ['MutableMapping'], 'tuple': ['Sequence'], 'list': ['MutableSequence'], 'complex': ['Complex'], 'float': ['Real'], 'int': ['Integral'], 'bool': ['int'], 'str': ['Sequence'], 'basestring': ['Sequence'], 'bytes': ['ByteString'], 'range': ['Sequence'], 'bytearray': ['ByteString', 'MutableSequence'], 'memoryview': ['Sequence'], 'bytearray_iterator': ['Iterator'], 'dict_keys': ['KeysView'], 'dict_items': ['ItemsView'], 'dict_values': ['ValuesView'], 'dict_keyiterator': ['Iterator'], 'dict_valueiterator': ['Iterator'], 'dict_itemiterator': ['Iterator'], 'list_iterator': ['Iterator'], 'list_reverseiterator': ['Iterator'], 'range_iterator': ['Iterator'], 'longrange_iterator': ['Iterator'], 'set_iterator': ['Iterator'], 'tuple_iterator': ['Iterator'], 'str_iterator': ['Iterator'], 'zip_iterator': ['Iterator'], 'bytes_iterator': ['Iterator'], 'mappingproxy': ['Mapping'], 'generator': ['Generator'], 'async_generator': ['AsyncGenerator'], 'coroutine': ['Coroutine']}

def GetSuperClasses():
    if False:
        i = 10
        return i + 15
    'Get a Python type hierarchy mapping.\n\n  This generates a dictionary that can be used to look up the bases of\n  a type in the abstract base class hierarchy.\n\n  Returns:\n    A dictionary mapping a type, as string, to a list of base types (also\n    as strings). E.g. "float" -> ["Real"].\n  '
    return SUPERCLASSES.copy()

def GetSubClasses():
    if False:
        i = 10
        return i + 15
    'Get a reverse Python type hierarchy mapping.\n\n  This generates a dictionary that can be used to look up the (known)\n  subclasses of a type in the abstract base class hierarchy.\n\n  Returns:\n    A dictionary mapping a type, as string, to a list of direct\n    subclasses (also as strings).\n    E.g. "Sized" -> ["Set", "Mapping", "MappingView", "Sequence"].\n  '
    return utils.invert_dict(GetSuperClasses())