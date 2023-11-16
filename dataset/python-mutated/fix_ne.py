"""Fixer that turns <> into !=."""
from .. import pytree
from ..pgen2 import token
from .. import fixer_base

class FixNe(fixer_base.BaseFix):
    _accept_type = token.NOTEQUAL

    def match(self, node):
        if False:
            i = 10
            return i + 15
        return node.value == '<>'

    def transform(self, node, results):
        if False:
            return 10
        new = pytree.Leaf(token.NOTEQUAL, '!=', prefix=node.prefix)
        return new