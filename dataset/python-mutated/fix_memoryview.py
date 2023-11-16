u"""
Fixer for memoryview(s) -> buffer(s).
Explicit because some memoryview methods are invalid on buffer objects.
"""
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name

class FixMemoryview(fixer_base.BaseFix):
    explicit = True
    PATTERN = u"\n              power< name='memoryview' trailer< '(' [any] ')' >\n              rest=any* >\n              "

    def transform(self, node, results):
        if False:
            for i in range(10):
                print('nop')
        name = results[u'name']
        name.replace(Name(u'buffer', prefix=name.prefix))