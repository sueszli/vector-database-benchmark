"""
Fixer for the cmp() function on Py2, which was removed in Py3.

Adds this import line::

    from past.builtins import cmp

if cmp() is called in the code.
"""
from __future__ import unicode_literals
from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top
expression = "name='cmp'"

class FixCmp(fixer_base.BaseFix):
    BM_compatible = True
    run_order = 9
    PATTERN = "\n              power<\n                 ({0}) trailer< '(' args=[any] ')' >\n              rest=any* >\n              ".format(expression)

    def transform(self, node, results):
        if False:
            i = 10
            return i + 15
        name = results['name']
        touch_import_top(u'past.builtins', name.value, node)