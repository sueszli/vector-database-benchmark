"""Support for wildcard pattern matching in object inspection.

Authors
-------
- Jörgen Stenarson <jorgen.stenarson@bostream.nu>
- Thomas Kluyver
"""
import re
import types
from IPython.utils.dir2 import dir2

def create_typestr2type_dicts(dont_include_in_type2typestr=['lambda']):
    if False:
        return 10
    "Return dictionaries mapping lower case typename (e.g. 'tuple') to type\n    objects from the types package, and vice versa."
    typenamelist = [tname for tname in dir(types) if tname.endswith('Type')]
    (typestr2type, type2typestr) = ({}, {})
    for tname in typenamelist:
        name = tname[:-4].lower()
        obj = getattr(types, tname)
        typestr2type[name] = obj
        if name not in dont_include_in_type2typestr:
            type2typestr[obj] = name
    return (typestr2type, type2typestr)
(typestr2type, type2typestr) = create_typestr2type_dicts()

def is_type(obj, typestr_or_type):
    if False:
        print('Hello World!')
    "is_type(obj, typestr_or_type) verifies if obj is of a certain type. It\n    can take strings or actual python types for the second argument, i.e.\n    'tuple'<->TupleType. 'all' matches all types.\n\n    TODO: Should be extended for choosing more than one type."
    if typestr_or_type == 'all':
        return True
    if type(typestr_or_type) == type:
        test_type = typestr_or_type
    else:
        test_type = typestr2type.get(typestr_or_type, False)
    if test_type:
        return isinstance(obj, test_type)
    return False

def show_hidden(str, show_all=False):
    if False:
        return 10
    'Return true for strings starting with single _ if show_all is true.'
    return show_all or str.startswith('__') or (not str.startswith('_'))

def dict_dir(obj):
    if False:
        print('Hello World!')
    "Produce a dictionary of an object's attributes. Builds on dir2 by\n    checking that a getattr() call actually succeeds."
    ns = {}
    for key in dir2(obj):
        try:
            ns[key] = getattr(obj, key)
        except AttributeError:
            pass
    return ns

def filter_ns(ns, name_pattern='*', type_pattern='all', ignore_case=True, show_all=True):
    if False:
        print('Hello World!')
    'Filter a namespace dictionary by name pattern and item type.'
    pattern = name_pattern.replace('*', '.*').replace('?', '.')
    if ignore_case:
        reg = re.compile(pattern + '$', re.I)
    else:
        reg = re.compile(pattern + '$')
    return dict(((key, obj) for (key, obj) in ns.items() if reg.match(key) and show_hidden(key, show_all) and is_type(obj, type_pattern)))

def list_namespace(namespace, type_pattern, filter, ignore_case=False, show_all=False):
    if False:
        i = 10
        return i + 15
    'Return dictionary of all objects in a namespace dictionary that match\n    type_pattern and filter.'
    pattern_list = filter.split('.')
    if len(pattern_list) == 1:
        return filter_ns(namespace, name_pattern=pattern_list[0], type_pattern=type_pattern, ignore_case=ignore_case, show_all=show_all)
    else:
        filtered = filter_ns(namespace, name_pattern=pattern_list[0], type_pattern='all', ignore_case=ignore_case, show_all=show_all)
        results = {}
        for (name, obj) in filtered.items():
            ns = list_namespace(dict_dir(obj), type_pattern, '.'.join(pattern_list[1:]), ignore_case=ignore_case, show_all=show_all)
            for (inner_name, inner_obj) in ns.items():
                results['%s.%s' % (name, inner_name)] = inner_obj
        return results