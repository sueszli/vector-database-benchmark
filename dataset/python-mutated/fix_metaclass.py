"""Fixer for __metaclass__ = X -> (metaclass=X) methods.

   The various forms of classef (inherits nothing, inherits once, inherits
   many) don't parse the same in the CST so we look at ALL classes for
   a __metaclass__ and if we find one normalize the inherits to all be
   an arglist.

   For one-liner classes ('class X: pass') there is no indent/dedent so
   we normalize those into having a suite.

   Moving the __metaclass__ into the classdef can also cause the class
   body to be empty so there is some special casing for that as well.

   This fixer also tries very hard to keep original indenting and spacing
   in all those corner cases.

"""
from .. import fixer_base
from ..pygram import token
from ..fixer_util import syms, Node, Leaf

def has_metaclass(parent):
    if False:
        i = 10
        return i + 15
    " we have to check the cls_node without changing it.\n        There are two possibilities:\n          1)  clsdef => suite => simple_stmt => expr_stmt => Leaf('__meta')\n          2)  clsdef => simple_stmt => expr_stmt => Leaf('__meta')\n    "
    for node in parent.children:
        if node.type == syms.suite:
            return has_metaclass(node)
        elif node.type == syms.simple_stmt and node.children:
            expr_node = node.children[0]
            if expr_node.type == syms.expr_stmt and expr_node.children:
                left_side = expr_node.children[0]
                if isinstance(left_side, Leaf) and left_side.value == '__metaclass__':
                    return True
    return False

def fixup_parse_tree(cls_node):
    if False:
        while True:
            i = 10
    " one-line classes don't get a suite in the parse tree so we add\n        one to normalize the tree\n    "
    for node in cls_node.children:
        if node.type == syms.suite:
            return
    for (i, node) in enumerate(cls_node.children):
        if node.type == token.COLON:
            break
    else:
        raise ValueError("No class suite and no ':'!")
    suite = Node(syms.suite, [])
    while cls_node.children[i + 1:]:
        move_node = cls_node.children[i + 1]
        suite.append_child(move_node.clone())
        move_node.remove()
    cls_node.append_child(suite)
    node = suite

def fixup_simple_stmt(parent, i, stmt_node):
    if False:
        i = 10
        return i + 15
    ' if there is a semi-colon all the parts count as part of the same\n        simple_stmt.  We just want the __metaclass__ part so we move\n        everything after the semi-colon into its own simple_stmt node\n    '
    for (semi_ind, node) in enumerate(stmt_node.children):
        if node.type == token.SEMI:
            break
    else:
        return
    node.remove()
    new_expr = Node(syms.expr_stmt, [])
    new_stmt = Node(syms.simple_stmt, [new_expr])
    while stmt_node.children[semi_ind:]:
        move_node = stmt_node.children[semi_ind]
        new_expr.append_child(move_node.clone())
        move_node.remove()
    parent.insert_child(i, new_stmt)
    new_leaf1 = new_stmt.children[0].children[0]
    old_leaf1 = stmt_node.children[0].children[0]
    new_leaf1.prefix = old_leaf1.prefix

def remove_trailing_newline(node):
    if False:
        print('Hello World!')
    if node.children and node.children[-1].type == token.NEWLINE:
        node.children[-1].remove()

def find_metas(cls_node):
    if False:
        for i in range(10):
            print('nop')
    for node in cls_node.children:
        if node.type == syms.suite:
            break
    else:
        raise ValueError('No class suite!')
    for (i, simple_node) in list(enumerate(node.children)):
        if simple_node.type == syms.simple_stmt and simple_node.children:
            expr_node = simple_node.children[0]
            if expr_node.type == syms.expr_stmt and expr_node.children:
                left_node = expr_node.children[0]
                if isinstance(left_node, Leaf) and left_node.value == '__metaclass__':
                    fixup_simple_stmt(node, i, simple_node)
                    remove_trailing_newline(simple_node)
                    yield (node, i, simple_node)

def fixup_indent(suite):
    if False:
        return 10
    ' If an INDENT is followed by a thing with a prefix then nuke the prefix\n        Otherwise we get in trouble when removing __metaclass__ at suite start\n    '
    kids = suite.children[::-1]
    while kids:
        node = kids.pop()
        if node.type == token.INDENT:
            break
    while kids:
        node = kids.pop()
        if isinstance(node, Leaf) and node.type != token.DEDENT:
            if node.prefix:
                node.prefix = ''
            return
        else:
            kids.extend(node.children[::-1])

class FixMetaclass(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = '\n    classdef<any*>\n    '

    def transform(self, node, results):
        if False:
            i = 10
            return i + 15
        if not has_metaclass(node):
            return
        fixup_parse_tree(node)
        last_metaclass = None
        for (suite, i, stmt) in find_metas(node):
            last_metaclass = stmt
            stmt.remove()
        text_type = node.children[0].type
        if len(node.children) == 7:
            if node.children[3].type == syms.arglist:
                arglist = node.children[3]
            else:
                parent = node.children[3].clone()
                arglist = Node(syms.arglist, [parent])
                node.set_child(3, arglist)
        elif len(node.children) == 6:
            arglist = Node(syms.arglist, [])
            node.insert_child(3, arglist)
        elif len(node.children) == 4:
            arglist = Node(syms.arglist, [])
            node.insert_child(2, Leaf(token.RPAR, ')'))
            node.insert_child(2, arglist)
            node.insert_child(2, Leaf(token.LPAR, '('))
        else:
            raise ValueError('Unexpected class definition')
        meta_txt = last_metaclass.children[0].children[0]
        meta_txt.value = 'metaclass'
        orig_meta_prefix = meta_txt.prefix
        if arglist.children:
            arglist.append_child(Leaf(token.COMMA, ','))
            meta_txt.prefix = ' '
        else:
            meta_txt.prefix = ''
        expr_stmt = last_metaclass.children[0]
        assert expr_stmt.type == syms.expr_stmt
        expr_stmt.children[1].prefix = ''
        expr_stmt.children[2].prefix = ''
        arglist.append_child(last_metaclass)
        fixup_indent(suite)
        if not suite.children:
            suite.remove()
            pass_leaf = Leaf(text_type, 'pass')
            pass_leaf.prefix = orig_meta_prefix
            node.append_child(pass_leaf)
            node.append_child(Leaf(token.NEWLINE, '\n'))
        elif len(suite.children) > 1 and (suite.children[-2].type == token.INDENT and suite.children[-1].type == token.DEDENT):
            pass_leaf = Leaf(text_type, 'pass')
            suite.insert_child(-1, pass_leaf)
            suite.insert_child(-1, Leaf(token.NEWLINE, '\n'))