"""Remove __future__ imports

from __future__ import foo is replaced with an empty line.
"""
from .. import fixer_base
from ..fixer_util import BlankLine

class FixFuture(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = 'import_from< \'from\' module_name="__future__" \'import\' any >'
    run_order = 10

    def transform(self, node, results):
        if False:
            return 10
        new = BlankLine()
        new.prefix = node.prefix
        return new