from collections import defaultdict
from functools import total_ordering
import enum

class Conversion(enum.IntEnum):
    """
    A conversion kind from one type to the other.  The enum members
    are ordered from stricter to looser.
    """
    exact = 1
    promote = 2
    safe = 3
    unsafe = 4
    nil = 99

class CastSet(object):
    """A set of casting rules.

    There is at most one rule per target type.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._rels = {}

    def insert(self, to, rel):
        if False:
            print('Hello World!')
        old = self.get(to)
        setrel = min(rel, old)
        self._rels[to] = setrel
        return old != setrel

    def items(self):
        if False:
            return 10
        return self._rels.items()

    def get(self, item):
        if False:
            i = 10
            return i + 15
        return self._rels.get(item, Conversion.nil)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._rels)

    def __repr__(self):
        if False:
            return 10
        body = ['{rel}({ty})'.format(rel=rel, ty=ty) for (ty, rel) in self._rels.items()]
        return '{' + ', '.join(body) + '}'

    def __contains__(self, item):
        if False:
            return 10
        return item in self._rels

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._rels.keys())

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return self._rels[item]

class TypeGraph(object):
    """A graph that maintains the casting relationship of all types.

    This simplifies the definition of casting rules by automatically
    propagating the rules.
    """

    def __init__(self, callback=None):
        if False:
            while True:
                i = 10
        '\n        Args\n        ----\n        - callback: callable or None\n            It is called for each new casting rule with\n            (from_type, to_type, castrel).\n        '
        assert callback is None or callable(callback)
        self._forwards = defaultdict(CastSet)
        self._backwards = defaultdict(set)
        self._callback = callback

    def get(self, ty):
        if False:
            return 10
        return self._forwards[ty]

    def propagate(self, a, b, baserel):
        if False:
            return 10
        backset = self._backwards[a]
        for child in self._forwards[b]:
            rel = max(baserel, self._forwards[b][child])
            if a != child:
                if self._forwards[a].insert(child, rel):
                    self._callback(a, child, rel)
                self._backwards[child].add(a)
            for backnode in backset:
                if backnode != child:
                    backrel = max(rel, self._forwards[backnode][a])
                    if self._forwards[backnode].insert(child, backrel):
                        self._callback(backnode, child, backrel)
                    self._backwards[child].add(backnode)
        for child in self._backwards[a]:
            rel = max(baserel, self._forwards[child][a])
            if b != child:
                if self._forwards[child].insert(b, rel):
                    self._callback(child, b, rel)
                self._backwards[b].add(child)

    def insert_rule(self, a, b, rel):
        if False:
            print('Hello World!')
        self._forwards[a].insert(b, rel)
        self._callback(a, b, rel)
        self._backwards[b].add(a)
        self.propagate(a, b, rel)

    def promote(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        self.insert_rule(a, b, Conversion.promote)

    def safe(self, a, b):
        if False:
            i = 10
            return i + 15
        self.insert_rule(a, b, Conversion.safe)

    def unsafe(self, a, b):
        if False:
            return 10
        self.insert_rule(a, b, Conversion.unsafe)