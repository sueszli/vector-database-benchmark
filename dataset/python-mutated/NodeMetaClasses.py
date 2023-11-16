""" Node meta classes.

This provides meta classes for nodes, currently only one. These do all kinds
of checks, and add methods automatically.

"""
from abc import ABCMeta
from nuitka.__past__ import intern
from nuitka.Errors import NuitkaAssumptionError, NuitkaNodeDesignError
from nuitka.PythonVersions import python_version

def _checkBases(name, bases):
    if False:
        i = 10
        return i + 15
    assert len(bases) == len(set(bases)), (name, bases)
    last_mixin = None
    for base in bases:
        base_name = base.__name__
        is_mixin = base_name.endswith('Mixin')
        if is_mixin and last_mixin is False:
            raise NuitkaNodeDesignError(name, 'Mixins must come first in base classes.', bases)
        last_mixin = is_mixin
        if base is not object and '__slots__' not in base.__dict__:
            raise NuitkaNodeDesignError(name, 'All bases must set __slots__.', base)

class NodeCheckMetaClass(ABCMeta):
    kinds = {}

    def __new__(mcs, name, bases, dictionary):
        if False:
            for i in range(10):
                print('nop')
        _checkBases(name, bases)
        if '__slots__' not in dictionary:
            dictionary['__slots__'] = ()
        if 'named_children' in dictionary:
            assert type(dictionary['named_children']) is tuple
            dictionary['__slots__'] += tuple((intern('subnode_' + named_child.split('|', 1)[0]) for named_child in dictionary['named_children']))
        if 'nice_children' in dictionary:
            assert type(dictionary['nice_children']) is tuple
            assert len(dictionary['nice_children']) == len(dictionary['named_children'])
            dictionary['nice_children_dict'] = dict(((intern(named_child.split('|', 1)[0]), nice_name) for (named_child, nice_name) in zip(dictionary['named_children'], dictionary['nice_children'])))
        if 'node_attributes' in dictionary:
            dictionary['__slots__'] += dictionary['node_attributes']
        assert len(dictionary['__slots__']) == len(set(dictionary['__slots__'])), dictionary['__slots__']
        if 'python_version_spec' in dictionary:
            condition = '%s %s' % (hex(python_version), dictionary['python_version_spec'])
            if not eval(condition):

                def __init__(self, *args, **kwargs):
                    if False:
                        print('Hello World!')
                    raise NuitkaAssumptionError(name, 'assumption violated', condition)
                dictionary['__init__'] = __init__
        if 'checker' in dictionary:
            dictionary['checker'] = staticmethod(dictionary['checker'])
        return ABCMeta.__new__(mcs, name, bases, dictionary)

    def __init__(cls, name, bases, dictionary):
        if False:
            while True:
                i = 10
        if not name.endswith(('Base', 'Mixin')):
            if 'kind' not in dictionary:
                raise NuitkaNodeDesignError(name, "Must provide class variable 'kind'")
            kind = dictionary['kind']
            assert type(kind) is str, name
            if kind in NodeCheckMetaClass.kinds and (not 'replaces' in dictionary):
                raise NuitkaNodeDesignError(name, "Duplicate nodes for kind '%s'" % kind)
            NodeCheckMetaClass.kinds[kind] = cls
            NodeCheckMetaClass.kinds[name] = cls
            kind_to_name_part = ''.join([x.capitalize() for x in kind.split('_')])
            assert name.endswith(kind_to_name_part), (name, kind_to_name_part)
            checker_method = 'is' + kind_to_name_part

            def checkKind(self):
                if False:
                    return 10
                return self.kind == kind
            from .NodeBases import NodeBase
            if not hasattr(NodeBase, checker_method):
                setattr(NodeBase, checker_method, checkKind)
        ABCMeta.__init__(cls, name, bases, dictionary)
NodeMetaClassBase = NodeCheckMetaClass('NodeMetaClassBase', (object,), {'__slots__': ()})