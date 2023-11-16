"""
Based on fix_next.py by Collin Winter.

Replaces it.next() -> next(it), per PEP 3114.

Unlike fix_next.py, this fixer doesn't replace the name of a next method with __next__,
which would break Python 2 compatibility without further help from fixers in
stage 2.
"""
from lib2to3.pgen2 import token
from lib2to3.pygram import python_symbols as syms
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, Call, find_binding
bind_warning = 'Calls to builtin next() possibly shadowed by global binding'

class FixNextCall(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "\n    power< base=any+ trailer< '.' attr='next' > trailer< '(' ')' > >\n    |\n    power< head=any+ trailer< '.' attr='next' > not trailer< '(' ')' > >\n    |\n    global=global_stmt< 'global' any* 'next' any* >\n    "
    order = 'pre'

    def start_tree(self, tree, filename):
        if False:
            return 10
        super(FixNextCall, self).start_tree(tree, filename)
        n = find_binding('next', tree)
        if n:
            self.warning(n, bind_warning)
            self.shadowed_next = True
        else:
            self.shadowed_next = False

    def transform(self, node, results):
        if False:
            i = 10
            return i + 15
        assert results
        base = results.get('base')
        attr = results.get('attr')
        name = results.get('name')
        if base:
            if self.shadowed_next:
                pass
            else:
                base = [n.clone() for n in base]
                base[0].prefix = ''
                node.replace(Call(Name('next', prefix=node.prefix), base))
        elif name:
            pass
        elif attr:
            if is_assign_target(node):
                head = results['head']
                if ''.join([str(n) for n in head]).strip() == '__builtin__':
                    self.warning(node, bind_warning)
                return
        elif 'global' in results:
            self.warning(node, bind_warning)
            self.shadowed_next = True

def is_assign_target(node):
    if False:
        i = 10
        return i + 15
    assign = find_assign(node)
    if assign is None:
        return False
    for child in assign.children:
        if child.type == token.EQUAL:
            return False
        elif is_subtree(child, node):
            return True
    return False

def find_assign(node):
    if False:
        for i in range(10):
            print('nop')
    if node.type == syms.expr_stmt:
        return node
    if node.type == syms.simple_stmt or node.parent is None:
        return None
    return find_assign(node.parent)

def is_subtree(root, node):
    if False:
        while True:
            i = 10
    if root == node:
        return True
    return any((is_subtree(c, node) for c in root.children))