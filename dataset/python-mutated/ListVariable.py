"""engine.SCons.Variables.ListVariable

This file defines the option type for SCons implementing 'lists'.

A 'list' option may either be 'all', 'none' or a list of names
separated by comma. After the option has been processed, the option
value holds either the named list elements, all list elements or no
list elements at all.

Usage example::

    list_of_libs = Split('x11 gl qt ical')

    opts = Variables()
    opts.Add(ListVariable('shared',
                      'libraries to build as shared libraries',
                      'all',
                      elems = list_of_libs))
    ...
    for lib in list_of_libs:
     if lib in env['shared']:
         env.SharedObject(...)
     else:
         env.Object(...)
"""
__revision__ = 'src/engine/SCons/Variables/ListVariable.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__all__ = ['ListVariable']
import collections
import SCons.Util

class _ListVariable(collections.UserList):

    def __init__(self, initlist=[], allowedElems=[]):
        if False:
            return 10
        collections.UserList.__init__(self, [_f for _f in initlist if _f])
        self.allowedElems = sorted(allowedElems)

    def __cmp__(self, other):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def __ge__(self, other):
        if False:
            return 10
        raise NotImplementedError

    def __gt__(self, other):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __le__(self, other):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if len(self) == 0:
            return 'none'
        self.data.sort()
        if self.data == self.allowedElems:
            return 'all'
        else:
            return ','.join(self)

    def prepare_to_store(self):
        if False:
            i = 10
            return i + 15
        return self.__str__()

def _converter(val, allowedElems, mapdict):
    if False:
        print('Hello World!')
    '\n    '
    if val == 'none':
        val = []
    elif val == 'all':
        val = allowedElems
    else:
        val = [_f for _f in val.split(',') if _f]
        val = [mapdict.get(v, v) for v in val]
        notAllowed = [v for v in val if v not in allowedElems]
        if notAllowed:
            raise ValueError('Invalid value(s) for option: %s' % ','.join(notAllowed))
    return _ListVariable(val, allowedElems)

def ListVariable(key, help, default, names, map={}):
    if False:
        while True:
            i = 10
    "\n    The input parameters describe a 'package list' option, thus they\n    are returned with the correct converter and validator appended. The\n    result is usable for input to opts.Add() .\n\n    A 'package list' option may either be 'all', 'none' or a list of\n    package names (separated by space).\n    "
    names_str = 'allowed names: %s' % ' '.join(names)
    if SCons.Util.is_List(default):
        default = ','.join(default)
    help = '\n    '.join((help, '(all|none|comma-separated list of names)', names_str))
    return (key, help, default, None, lambda val: _converter(val, names, map))