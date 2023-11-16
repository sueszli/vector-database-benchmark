"""
For the ``future`` package.

Adds this import line:

    from __future__ import division

at the top and changes any old-style divisions to be calls to
past.utils.old_div so the code runs as before on Py2.6/2.7 and has the same
behaviour on Py3.

If "from __future__ import division" is already in effect, this fixer does
nothing.
"""
import re
from lib2to3.fixer_util import Leaf, Node, Comma
from lib2to3 import fixer_base
from libfuturize.fixer_util import token, future_import, touch_import_top, wrap_in_fn_call

def match_division(node):
    if False:
        for i in range(10):
            print('nop')
    u'\n    __future__.division redefines the meaning of a single slash for division,\n    so we match that and only that.\n    '
    slash = token.SLASH
    return node.type == slash and (not node.next_sibling.type == slash) and (not node.prev_sibling.type == slash)
const_re = re.compile('^[0-9]*[.][0-9]*$')

def is_floaty(node):
    if False:
        for i in range(10):
            print('nop')
    return _is_floaty(node.prev_sibling) or _is_floaty(node.next_sibling)

def _is_floaty(expr):
    if False:
        print('Hello World!')
    if isinstance(expr, list):
        expr = expr[0]
    if isinstance(expr, Leaf):
        return const_re.match(expr.value)
    elif isinstance(expr, Node):
        if isinstance(expr.children[0], Leaf):
            return expr.children[0].value == u'float'
    return False

class FixDivisionSafe(fixer_base.BaseFix):
    run_order = 4
    _accept_type = token.SLASH
    PATTERN = "\n    term<(not('/') any)+ '/' ((not('/') any))>\n    "

    def start_tree(self, tree, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Skip this fixer if "__future__.division" is already imported.\n        '
        super(FixDivisionSafe, self).start_tree(tree, name)
        self.skip = 'division' in tree.future_features

    def match(self, node):
        if False:
            return 10
        u'\n        Since the tree needs to be fixed once and only once if and only if it\n        matches, we can start discarding matches after the first.\n        '
        if node.type == self.syms.term:
            matched = False
            skip = False
            children = []
            for child in node.children:
                if skip:
                    skip = False
                    continue
                if match_division(child) and (not is_floaty(child)):
                    matched = True
                    children[0].prefix = u''
                    children = [wrap_in_fn_call('old_div', children + [Comma(), child.next_sibling.clone()], prefix=node.prefix)]
                    skip = True
                else:
                    children.append(child.clone())
            if matched:
                if hasattr(Node, 'fixers_applied'):
                    return Node(node.type, children, fixers_applied=node.fixers_applied)
                else:
                    return Node(node.type, children)
        return False

    def transform(self, node, results):
        if False:
            return 10
        if self.skip:
            return
        future_import(u'division', node)
        touch_import_top(u'past.utils', u'old_div', node)
        return results