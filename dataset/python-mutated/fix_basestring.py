"""Fixer for basestring -> str."""
from .. import fixer_base
from ..fixer_util import Name

class FixBasestring(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "'basestring'"

    def transform(self, node, results):
        if False:
            while True:
                i = 10
        return Name('str', prefix=node.prefix)