"""Utility functions, node construction macros, etc."""
from . import patcomp
from .pgen2 import token
from .pygram import python_symbols as syms
from .pytree import Leaf
from .pytree import Node

def KeywordArg(keyword, value):
    if False:
        while True:
            i = 10
    return Node(syms.argument, [keyword, Leaf(token.EQUAL, '='), value])

def LParen():
    if False:
        print('Hello World!')
    return Leaf(token.LPAR, '(')

def RParen():
    if False:
        for i in range(10):
            print('nop')
    return Leaf(token.RPAR, ')')

def Assign(target, source):
    if False:
        return 10
    'Build an assignment statement'
    if not isinstance(target, list):
        target = [target]
    if not isinstance(source, list):
        source.prefix = ' '
        source = [source]
    return Node(syms.atom, target + [Leaf(token.EQUAL, '=', prefix=' ')] + source)

def Name(name, prefix=None):
    if False:
        while True:
            i = 10
    'Return a NAME leaf'
    return Leaf(token.NAME, name, prefix=prefix)

def Attr(obj, attr):
    if False:
        i = 10
        return i + 15
    'A node tuple for obj.attr'
    return [obj, Node(syms.trailer, [Dot(), attr])]

def Comma():
    if False:
        return 10
    'A comma leaf'
    return Leaf(token.COMMA, ',')

def Dot():
    if False:
        for i in range(10):
            print('nop')
    'A period (.) leaf'
    return Leaf(token.DOT, '.')

def ArgList(args, lparen=LParen(), rparen=RParen()):
    if False:
        print('Hello World!')
    'A parenthesised argument list, used by Call()'
    node = Node(syms.trailer, [lparen.clone(), rparen.clone()])
    if args:
        node.insert_child(1, Node(syms.arglist, args))
    return node

def Call(func_name, args=None, prefix=None):
    if False:
        i = 10
        return i + 15
    'A function call'
    node = Node(syms.power, [func_name, ArgList(args)])
    if prefix is not None:
        node.prefix = prefix
    return node

def Newline():
    if False:
        print('Hello World!')
    'A newline literal'
    return Leaf(token.NEWLINE, '\n')

def BlankLine():
    if False:
        return 10
    'A blank line'
    return Leaf(token.NEWLINE, '')

def Number(n, prefix=None):
    if False:
        for i in range(10):
            print('nop')
    return Leaf(token.NUMBER, n, prefix=prefix)

def Subscript(index_node):
    if False:
        return 10
    'A numeric or string subscript'
    return Node(syms.trailer, [Leaf(token.LBRACE, '['), index_node, Leaf(token.RBRACE, ']')])

def String(string, prefix=None):
    if False:
        for i in range(10):
            print('nop')
    'A string leaf'
    return Leaf(token.STRING, string, prefix=prefix)

def ListComp(xp, fp, it, test=None):
    if False:
        for i in range(10):
            print('nop')
    'A list comprehension of the form [xp for fp in it if test].\n\n  If test is None, the "if test" part is omitted.\n  '
    xp.prefix = ''
    fp.prefix = ' '
    it.prefix = ' '
    for_leaf = Leaf(token.NAME, 'for')
    for_leaf.prefix = ' '
    in_leaf = Leaf(token.NAME, 'in')
    in_leaf.prefix = ' '
    inner_args = [for_leaf, fp, in_leaf, it]
    if test:
        test.prefix = ' '
        if_leaf = Leaf(token.NAME, 'if')
        if_leaf.prefix = ' '
        inner_args.append(Node(syms.comp_if, [if_leaf, test]))
    inner = Node(syms.listmaker, [xp, Node(syms.comp_for, inner_args)])
    return Node(syms.atom, [Leaf(token.LBRACE, '['), inner, Leaf(token.RBRACE, ']')])

def FromImport(package_name, name_leafs):
    if False:
        i = 10
        return i + 15
    ' Return an import statement in the form:\n\n       from package import name_leafs\n  '
    for leaf in name_leafs:
        leaf.remove()
    children = [Leaf(token.NAME, 'from'), Leaf(token.NAME, package_name, prefix=' '), Leaf(token.NAME, 'import', prefix=' '), Node(syms.import_as_names, name_leafs)]
    imp = Node(syms.import_from, children)
    return imp

def ImportAndCall(node, results, names):
    if False:
        print('Hello World!')
    'Returns an import statement and calls a method of the module:\n\n      import module\n      module.name()\n  '
    obj = results['obj'].clone()
    if obj.type == syms.arglist:
        newarglist = obj.clone()
    else:
        newarglist = Node(syms.arglist, [obj.clone()])
    after = results['after']
    if after:
        after = [n.clone() for n in after]
    new = Node(syms.power, Attr(Name(names[0]), Name(names[1])) + [Node(syms.trailer, [results['lpar'].clone(), newarglist, results['rpar'].clone()])] + after)
    new.prefix = node.prefix
    return new

def is_tuple(node):
    if False:
        return 10
    'Does the node represent a tuple literal?'
    if isinstance(node, Node) and node.children == [LParen(), RParen()]:
        return True
    return isinstance(node, Node) and len(node.children) == 3 and isinstance(node.children[0], Leaf) and isinstance(node.children[1], Node) and isinstance(node.children[2], Leaf) and (node.children[0].value == '(') and (node.children[2].value == ')')

def is_list(node):
    if False:
        print('Hello World!')
    'Does the node represent a list literal?'
    return isinstance(node, Node) and len(node.children) > 1 and isinstance(node.children[0], Leaf) and isinstance(node.children[-1], Leaf) and (node.children[0].value == '[') and (node.children[-1].value == ']')

def parenthesize(node):
    if False:
        print('Hello World!')
    return Node(syms.atom, [LParen(), node, RParen()])
consuming_calls = {'sorted', 'list', 'set', 'any', 'all', 'tuple', 'sum', 'min', 'max', 'enumerate'}

def attr_chain(obj, attr):
    if False:
        return 10
    'Follow an attribute chain.\n\n  If you have a chain of objects where a.foo -> b, b.foo-> c, etc, use this to\n  iterate over all objects in the chain. Iteration is terminated by getattr(x,\n  attr) is None.\n\n  Args:\n      obj: the starting object\n      attr: the name of the chaining attribute\n\n  Yields:\n      Each successive object in the chain.\n  '
    next = getattr(obj, attr)
    while next:
        yield next
        next = getattr(next, attr)
p0 = "for_stmt< 'for' any 'in' node=any ':' any* >\n        | comp_for< 'for' any 'in' node=any any* >\n     "
p1 = "\npower<\n    ( 'iter' | 'list' | 'tuple' | 'sorted' | 'set' | 'sum' |\n      'any' | 'all' | 'enumerate' | (any* trailer< '.' 'join' >) )\n    trailer< '(' node=any ')' >\n    any*\n>\n"
p2 = "\npower<\n    ( 'sorted' | 'enumerate' )\n    trailer< '(' arglist<node=any any*> ')' >\n    any*\n>\n"
pats_built = False

def in_special_context(node):
    if False:
        return 10
    " Returns true if node is in an environment where all that is required\n      of it is being iterable (ie, it doesn't matter if it returns a list\n      or an iterator).\n      See test_map_nochange in test_fixers.py for some examples and tests.\n  "
    global p0, p1, p2, pats_built
    if not pats_built:
        p0 = patcomp.compile_pattern(p0)
        p1 = patcomp.compile_pattern(p1)
        p2 = patcomp.compile_pattern(p2)
        pats_built = True
    patterns = [p0, p1, p2]
    for (pattern, parent) in zip(patterns, attr_chain(node, 'parent')):
        results = {}
        if pattern.match(parent, results) and results['node'] is node:
            return True
    return False

def is_probably_builtin(node):
    if False:
        print('Hello World!')
    "Check that something isn't an attribute or function name etc."
    prev = node.prev_sibling
    if prev is not None and prev.type == token.DOT:
        return False
    parent = node.parent
    if parent.type in (syms.funcdef, syms.classdef):
        return False
    if parent.type == syms.expr_stmt and parent.children[0] is node:
        return False
    if parent.type == syms.parameters or (parent.type == syms.typedargslist and (prev is not None and prev.type == token.COMMA or parent.children[0] is node)):
        return False
    return True

def find_indentation(node):
    if False:
        for i in range(10):
            print('nop')
    'Find the indentation of *node*.'
    while node is not None:
        if node.type == syms.suite and len(node.children) > 2:
            indent = node.children[1]
            if indent.type == token.INDENT:
                return indent.value
        node = node.parent
    return ''

def make_suite(node):
    if False:
        i = 10
        return i + 15
    if node.type == syms.suite:
        return node
    node = node.clone()
    (parent, node.parent) = (node.parent, None)
    suite = Node(syms.suite, [node])
    suite.parent = parent
    return suite

def find_root(node):
    if False:
        return 10
    'Find the top level namespace.'
    while node.type != syms.file_input:
        node = node.parent
        if not node:
            raise ValueError('root found before file_input node was found.')
    return node

def does_tree_import(package, name, node):
    if False:
        print('Hello World!')
    " Returns true if name is imported from package at the\n      top level of the tree which node belongs to.\n      To cover the case of an import like 'import foo', use\n      None for the package and 'foo' for the name.\n  "
    binding = find_binding(name, find_root(node), package)
    return bool(binding)

def is_import(node):
    if False:
        print('Hello World!')
    'Returns true if the node is an import statement.'
    return node.type in (syms.import_name, syms.import_from)

def touch_import(package, name, node):
    if False:
        return 10
    ' Works like `does_tree_import` but adds an import statement\n      if it was not imported. '

    def is_import_stmt(node):
        if False:
            return 10
        return node.type == syms.simple_stmt and node.children and is_import(node.children[0])
    root = find_root(node)
    if does_tree_import(package, name, root):
        return
    insert_pos = offset = 0
    for (idx, node) in enumerate(root.children):
        if not is_import_stmt(node):
            continue
        for (offset, node2) in enumerate(root.children[idx:]):
            if not is_import_stmt(node2):
                break
        insert_pos = idx + offset
        break
    if insert_pos == 0:
        for (idx, node) in enumerate(root.children):
            if node.type == syms.simple_stmt and node.children and (node.children[0].type == token.STRING):
                insert_pos = idx + 1
                break
    if package is None:
        import_ = Node(syms.import_name, [Leaf(token.NAME, 'import'), Leaf(token.NAME, name, prefix=' ')])
    else:
        import_ = FromImport(package, [Leaf(token.NAME, name, prefix=' ')])
    children = [import_, Newline()]
    root.insert_child(insert_pos, Node(syms.simple_stmt, children))
_def_syms = {syms.classdef, syms.funcdef}

def find_binding(name, node, package=None):
    if False:
        print('Hello World!')
    ' Returns the node which binds variable name, otherwise None.\n      If optional argument package is supplied, only imports will\n      be returned.\n      See test cases for examples.\n  '
    for child in node.children:
        ret = None
        if child.type == syms.for_stmt:
            if _find(name, child.children[1]):
                return child
            n = find_binding(name, make_suite(child.children[-1]), package)
            if n:
                ret = n
        elif child.type in (syms.if_stmt, syms.while_stmt):
            n = find_binding(name, make_suite(child.children[-1]), package)
            if n:
                ret = n
        elif child.type == syms.try_stmt:
            n = find_binding(name, make_suite(child.children[2]), package)
            if n:
                ret = n
            else:
                for (i, kid) in enumerate(child.children[3:]):
                    if kid.type == token.COLON and kid.value == ':':
                        n = find_binding(name, make_suite(child.children[i + 4]), package)
                        if n:
                            ret = n
        elif child.type in _def_syms and child.children[1].value == name:
            ret = child
        elif _is_import_binding(child, name, package):
            ret = child
        elif child.type == syms.simple_stmt:
            ret = find_binding(name, child, package)
        elif child.type == syms.expr_stmt:
            if _find(name, child.children[0]):
                ret = child
        if ret:
            if not package:
                return ret
            if is_import(ret):
                return ret
    return None
_block_syms = {syms.funcdef, syms.classdef, syms.trailer}

def _find(name, node):
    if False:
        return 10
    nodes = [node]
    while nodes:
        node = nodes.pop()
        if node.type > 256 and node.type not in _block_syms:
            nodes.extend(node.children)
        elif node.type == token.NAME and node.value == name:
            return node
    return None

def _is_import_binding(node, name, package=None):
    if False:
        i = 10
        return i + 15
    ' Will return node if node will import name, or node\n      will import * from package.  None is returned otherwise.\n      See test cases for examples.\n  '
    if node.type == syms.import_name and (not package):
        imp = node.children[1]
        if imp.type == syms.dotted_as_names:
            for child in imp.children:
                if child.type == syms.dotted_as_name:
                    if child.children[2].value == name:
                        return node
                elif child.type == token.NAME and child.value == name:
                    return node
        elif imp.type == syms.dotted_as_name:
            last = imp.children[-1]
            if last.type == token.NAME and last.value == name:
                return node
        elif imp.type == token.NAME and imp.value == name:
            return node
    elif node.type == syms.import_from:
        if package and str(node.children[1]).strip() != package:
            return None
        n = node.children[3]
        if package and _find('as', n):
            return None
        elif n.type == syms.import_as_names and _find(name, n):
            return node
        elif n.type == syms.import_as_name:
            child = n.children[2]
            if child.type == token.NAME and child.value == name:
                return node
        elif n.type == token.NAME and n.value == name:
            return node
        elif package and n.type == token.STAR:
            return node
    return None