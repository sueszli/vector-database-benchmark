"""PEP484 compatibility code."""
from pytype.pytd import base_visitor
from pytype.pytd import pytd
ALL_TYPING_NAMES = ['AbstractSet', 'AnyStr', 'AsyncGenerator', 'BinaryIO', 'ByteString', 'Callable', 'Container', 'Dict', 'FrozenSet', 'Generator', 'Generic', 'Hashable', 'IO', 'ItemsView', 'Iterable', 'Iterator', 'KeysView', 'List', 'Mapping', 'MappingView', 'Match', 'MutableMapping', 'MutableSequence', 'MutableSet', 'NamedTuple', 'Optional', 'Pattern', 'Reversible', 'Sequence', 'Set', 'Sized', 'SupportsAbs', 'SupportsFloat', 'SupportsInt', 'SupportsRound', 'TextIO', 'Tuple', 'Type', 'TypeVar', 'Union']
COMPAT_ITEMS = [('NoneType', 'bool'), ('None', 'bool'), ('int', 'float'), ('int', 'complex'), ('float', 'complex'), ('bytearray', 'bytes'), ('memoryview', 'bytes')]
TYPING_TO_BUILTIN = {t: t.lower() for t in ['List', 'Dict', 'Tuple', 'Set', 'FrozenSet', 'Generator', 'Type', 'Coroutine', 'AsyncGenerator']}
BUILTIN_TO_TYPING = {v: k for (k, v) in TYPING_TO_BUILTIN.items()}

class ConvertTypingToNative(base_visitor.Visitor):
    """Visitor for converting PEP 484 types to native representation."""

    def __init__(self, module):
        if False:
            return 10
        super().__init__()
        self.module = module

    def _GetModuleAndName(self, t):
        if False:
            while True:
                i = 10
        if t.name and '.' in t.name:
            return t.name.rsplit('.', 1)
        else:
            return (None, t.name)

    def _IsTyping(self, module):
        if False:
            print('Hello World!')
        return module == 'typing' or (module is None and self.module == 'typing')

    def _Convert(self, t):
        if False:
            while True:
                i = 10
        (module, name) = self._GetModuleAndName(t)
        if not module and name == 'None':
            return pytd.NamedType('NoneType')
        elif self._IsTyping(module):
            if name in TYPING_TO_BUILTIN:
                return pytd.NamedType(TYPING_TO_BUILTIN[name])
            elif name == 'Any':
                return pytd.AnythingType()
            else:
                return t
        else:
            return t

    def VisitClassType(self, t):
        if False:
            print('Hello World!')
        return self._Convert(t)

    def VisitNamedType(self, t):
        if False:
            print('Hello World!')
        return self._Convert(t)

    def VisitGenericType(self, t):
        if False:
            return 10
        (module, name) = self._GetModuleAndName(t)
        if self._IsTyping(module):
            if name == 'Intersection':
                return pytd.IntersectionType(t.parameters)
            elif name == 'Optional':
                return pytd.UnionType(t.parameters + (pytd.NamedType('NoneType'),))
            elif name == 'Union':
                return pytd.UnionType(t.parameters)
        return t

    def VisitCallableType(self, t):
        if False:
            return 10
        return self.VisitGenericType(t)

    def VisitTupleType(self, t):
        if False:
            return 10
        return self.VisitGenericType(t)

    def VisitClass(self, node):
        if False:
            for i in range(10):
                print('nop')
        if self.module == 'builtins':
            bases = []
            for (old_base, new_base) in zip(self.old_node.bases, node.bases):
                if self._GetModuleAndName(new_base)[1] == node.name:
                    bases.append(old_base)
                else:
                    bases.append(new_base)
            return node.Replace(bases=tuple(bases))
        else:
            return node