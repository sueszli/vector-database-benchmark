u"""
Fixer for print: from __future__ import print_function.
"""
from lib2to3 import fixer_base
from libfuturize.fixer_util import future_import

class FixPrintfunction(fixer_base.BaseFix):
    PATTERN = u"\n              power< 'print' trailer < '(' any* ')' > any* >\n              "

    def transform(self, node, results):
        if False:
            print('Hello World!')
        future_import(u'print_function', node)