"""Fixer that changes buffer(...) into memoryview(...)."""
from .. import fixer_base
from ..fixer_util import Name

class FixBuffer(fixer_base.BaseFix):
    BM_compatible = True
    explicit = True
    PATTERN = "\n              power< name='buffer' trailer< '(' [any] ')' > any* >\n              "

    def transform(self, node, results):
        if False:
            return 10
        name = results['name']
        name.replace(Name('memoryview', prefix=name.prefix))