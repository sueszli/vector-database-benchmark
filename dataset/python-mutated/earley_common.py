"""This module implements useful building blocks for the Earley parser
"""

class Item:
    """An Earley Item, the atom of the algorithm."""
    __slots__ = ('s', 'rule', 'ptr', 'start', 'is_complete', 'expect', 'previous', 'node', '_hash')

    def __init__(self, rule, ptr, start):
        if False:
            return 10
        self.is_complete = len(rule.expansion) == ptr
        self.rule = rule
        self.ptr = ptr
        self.start = start
        self.node = None
        if self.is_complete:
            self.s = rule.origin
            self.expect = None
            self.previous = rule.expansion[ptr - 1] if ptr > 0 and len(rule.expansion) else None
        else:
            self.s = (rule, ptr)
            self.expect = rule.expansion[ptr]
            self.previous = rule.expansion[ptr - 1] if ptr > 0 and len(rule.expansion) else None
        self._hash = hash((self.s, self.start))

    def advance(self):
        if False:
            while True:
                i = 10
        return Item(self.rule, self.ptr + 1, self.start)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self is other or (self.s == other.s and self.start == other.start)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return self._hash

    def __repr__(self):
        if False:
            return 10
        before = (expansion.name for expansion in self.rule.expansion[:self.ptr])
        after = (expansion.name for expansion in self.rule.expansion[self.ptr:])
        symbol = '{} ::= {}* {}'.format(self.rule.origin.name, ' '.join(before), ' '.join(after))
        return '%s (%d)' % (symbol, self.start)