"""Fixer for __nonzero__ -> __bool__ methods."""
from .. import fixer_base
from ..fixer_util import Name

class FixNonzero(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "\n    classdef< 'class' any+ ':'\n              suite< any*\n                     funcdef< 'def' name='__nonzero__'\n                              parameters< '(' NAME ')' > any+ >\n                     any* > >\n    "

    def transform(self, node, results):
        if False:
            for i in range(10):
                print('nop')
        name = results['name']
        new = Name('__bool__', prefix=name.prefix)
        name.replace(new)