""" Utility functions to add/remove/get grists.
    Grists are string enclosed in angle brackets (<>) that are used as prefixes. See Jam for more information.
"""
import re
import os
import bjam
from b2.exceptions import *
from b2.util import is_iterable_typed
__re_grist_and_value = re.compile('(<[^>]*>)(.*)')
__re_grist_content = re.compile('^<(.*)>$')
__re_backslash = re.compile('\\\\')

def to_seq(value):
    if False:
        print('Hello World!')
    ' If value is a sequence, returns it.\n        If it is a string, returns a sequence with value as its sole element.\n        '
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    else:
        return value

def replace_references_by_objects(manager, refs):
    if False:
        i = 10
        return i + 15
    objs = []
    for r in refs:
        objs.append(manager.get_object(r))
    return objs

def add_grist(features):
    if False:
        for i in range(10):
            print('nop')
    ' Transform a string by bracketing it with "<>". If already bracketed, does nothing.\n        features: one string or a sequence of strings\n        return: the gristed string, if features is a string, or a sequence of gristed strings, if features is a sequence\n    '
    assert is_iterable_typed(features, basestring) or isinstance(features, basestring)

    def grist_one(feature):
        if False:
            print('Hello World!')
        if feature[0] != '<' and feature[len(feature) - 1] != '>':
            return '<' + feature + '>'
        else:
            return feature
    if isinstance(features, str):
        return grist_one(features)
    else:
        return [grist_one(feature) for feature in features]

def replace_grist(features, new_grist):
    if False:
        return 10
    ' Replaces the grist of a string by a new one.\n        Returns the string with the new grist.\n    '
    assert is_iterable_typed(features, basestring) or isinstance(features, basestring)
    assert isinstance(new_grist, basestring)
    single_item = False
    if isinstance(features, str):
        features = [features]
        single_item = True
    result = []
    for feature in features:
        (grist, split, value) = feature.partition('>')
        if not value and (not split):
            value = grist
        result.append(new_grist + value)
    if single_item:
        return result[0]
    return result

def get_value(property):
    if False:
        for i in range(10):
            print('nop')
    ' Gets the value of a property, that is, the part following the grist, if any.\n    '
    assert is_iterable_typed(property, basestring) or isinstance(property, basestring)
    return replace_grist(property, '')

def get_grist(value):
    if False:
        print('Hello World!')
    ' Returns the grist of a string.\n        If value is a sequence, does it for every value and returns the result as a sequence.\n    '
    assert is_iterable_typed(value, basestring) or isinstance(value, basestring)

    def get_grist_one(name):
        if False:
            i = 10
            return i + 15
        split = __re_grist_and_value.match(name)
        if not split:
            return ''
        else:
            return split.group(1)
    if isinstance(value, str):
        return get_grist_one(value)
    else:
        return [get_grist_one(v) for v in value]

def ungrist(value):
    if False:
        return 10
    ' Returns the value without grist.\n        If value is a sequence, does it for every value and returns the result as a sequence.\n    '
    assert is_iterable_typed(value, basestring) or isinstance(value, basestring)

    def ungrist_one(value):
        if False:
            print('Hello World!')
        stripped = __re_grist_content.match(value)
        if not stripped:
            raise BaseException("in ungrist: '%s' is not of the form <.*>" % value)
        return stripped.group(1)
    if isinstance(value, str):
        return ungrist_one(value)
    else:
        return [ungrist_one(v) for v in value]

def replace_suffix(name, new_suffix):
    if False:
        i = 10
        return i + 15
    ' Replaces the suffix of name by new_suffix.\n        If no suffix exists, the new one is added.\n    '
    assert isinstance(name, basestring)
    assert isinstance(new_suffix, basestring)
    split = os.path.splitext(name)
    return split[0] + new_suffix

def forward_slashes(s):
    if False:
        i = 10
        return i + 15
    ' Converts all backslashes to forward slashes.\n    '
    assert isinstance(s, basestring)
    return s.replace('\\', '/')

def split_action_id(id):
    if False:
        return 10
    " Splits an id in the toolset and specific rule parts. E.g.\n        'gcc.compile.c++' returns ('gcc', 'compile.c++')\n    "
    assert isinstance(id, basestring)
    split = id.split('.', 1)
    toolset = split[0]
    name = ''
    if len(split) > 1:
        name = split[1]
    return (toolset, name)

def os_name():
    if False:
        i = 10
        return i + 15
    result = bjam.variable('OS')
    assert len(result) == 1
    return result[0]

def platform():
    if False:
        return 10
    return bjam.variable('OSPLAT')

def os_version():
    if False:
        return 10
    return bjam.variable('OSVER')

def on_windows():
    if False:
        while True:
            i = 10
    ' Returns true if running on windows, whether in cygwin or not.\n    '
    if bjam.variable('NT'):
        return True
    elif bjam.variable('UNIX'):
        uname = bjam.variable('JAMUNAME')
        if uname and uname[0].startswith('CYGWIN'):
            return True
    return False