"""Fixer for reduce().

Makes sure reduce() is imported from the functools module if reduce is
used in that module.
"""
from lib2to3 import fixer_base
from lib2to3.fixer_util import touch_import

class FixReduce(fixer_base.BaseFix):
    BM_compatible = True
    order = 'pre'
    PATTERN = "\n    power< 'reduce'\n        trailer< '('\n            arglist< (\n                (not(argument<any '=' any>) any ','\n                 not(argument<any '=' any>) any) |\n                (not(argument<any '=' any>) any ','\n                 not(argument<any '=' any>) any ','\n                 not(argument<any '=' any>) any)\n            ) >\n        ')' >\n    >\n    "

    def transform(self, node, results):
        if False:
            return 10
        touch_import('functools', 'reduce', node)