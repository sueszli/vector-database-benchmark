u"""
Fixer for division: from __future__ import division if needed
"""
from lib2to3 import fixer_base
from libfuturize.fixer_util import token, future_import

def match_division(node):
    if False:
        return 10
    u'\n    __future__.division redefines the meaning of a single slash for division,\n    so we match that and only that.\n    '
    slash = token.SLASH
    return node.type == slash and (not node.next_sibling.type == slash) and (not node.prev_sibling.type == slash)

class FixDivision(fixer_base.BaseFix):
    run_order = 4

    def match(self, node):
        if False:
            while True:
                i = 10
        u'\n        Since the tree needs to be fixed once and only once if and only if it\n        matches, then we can start discarding matches after we make the first.\n        '
        return match_division(node)

    def transform(self, node, results):
        if False:
            for i in range(10):
                print('nop')
        future_import(u'division', node)