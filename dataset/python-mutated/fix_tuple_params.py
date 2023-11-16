"""Fixer for function definitions with tuple parameters.

def func(((a, b), c), d):
    ...

    ->

def func(x, d):
    ((a, b), c) = x
    ...

It will also support lambdas:

    lambda (x, y): x + y -> lambda t: t[0] + t[1]

    # The parens are a syntax error in Python 3
    lambda (x): x + y -> lambda x: x + y
"""
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Assign, Name, Newline, Number, Subscript, syms

def is_docstring(stmt):
    if False:
        while True:
            i = 10
    return isinstance(stmt, pytree.Node) and stmt.children[0].type == token.STRING

class FixTupleParams(fixer_base.BaseFix):
    run_order = 4
    BM_compatible = True
    PATTERN = "\n              funcdef< 'def' any parameters< '(' args=any ')' >\n                       ['->' any] ':' suite=any+ >\n              |\n              lambda=\n              lambdef< 'lambda' args=vfpdef< '(' inner=any ')' >\n                       ':' body=any\n              >\n              "

    def transform(self, node, results):
        if False:
            print('Hello World!')
        if 'lambda' in results:
            return self.transform_lambda(node, results)
        new_lines = []
        suite = results['suite']
        args = results['args']
        if suite[0].children[1].type == token.INDENT:
            start = 2
            indent = suite[0].children[1].value
            end = Newline()
        else:
            start = 0
            indent = '; '
            end = pytree.Leaf(token.INDENT, '')

        def handle_tuple(tuple_arg, add_prefix=False):
            if False:
                while True:
                    i = 10
            n = Name(self.new_name())
            arg = tuple_arg.clone()
            arg.prefix = ''
            stmt = Assign(arg, n.clone())
            if add_prefix:
                n.prefix = ' '
            tuple_arg.replace(n)
            new_lines.append(pytree.Node(syms.simple_stmt, [stmt, end.clone()]))
        if args.type == syms.tfpdef:
            handle_tuple(args)
        elif args.type == syms.typedargslist:
            for (i, arg) in enumerate(args.children):
                if arg.type == syms.tfpdef:
                    handle_tuple(arg, add_prefix=i > 0)
        if not new_lines:
            return
        for line in new_lines:
            line.parent = suite[0]
        after = start
        if start == 0:
            new_lines[0].prefix = ' '
        elif is_docstring(suite[0].children[start]):
            new_lines[0].prefix = indent
            after = start + 1
        for line in new_lines:
            line.parent = suite[0]
        suite[0].children[after:after] = new_lines
        for i in range(after + 1, after + len(new_lines) + 1):
            suite[0].children[i].prefix = indent
        suite[0].changed()

    def transform_lambda(self, node, results):
        if False:
            i = 10
            return i + 15
        args = results['args']
        body = results['body']
        inner = simplify_args(results['inner'])
        if inner.type == token.NAME:
            inner = inner.clone()
            inner.prefix = ' '
            args.replace(inner)
            return
        params = find_params(args)
        to_index = map_to_index(params)
        tup_name = self.new_name(tuple_name(params))
        new_param = Name(tup_name, prefix=' ')
        args.replace(new_param.clone())
        for n in body.post_order():
            if n.type == token.NAME and n.value in to_index:
                subscripts = [c.clone() for c in to_index[n.value]]
                new = pytree.Node(syms.power, [new_param.clone()] + subscripts)
                new.prefix = n.prefix
                n.replace(new)

def simplify_args(node):
    if False:
        return 10
    if node.type in (syms.vfplist, token.NAME):
        return node
    elif node.type == syms.vfpdef:
        while node.type == syms.vfpdef:
            node = node.children[1]
        return node
    raise RuntimeError('Received unexpected node %s' % node)

def find_params(node):
    if False:
        for i in range(10):
            print('nop')
    if node.type == syms.vfpdef:
        return find_params(node.children[1])
    elif node.type == token.NAME:
        return node.value
    return [find_params(c) for c in node.children if c.type != token.COMMA]

def map_to_index(param_list, prefix=[], d=None):
    if False:
        return 10
    if d is None:
        d = {}
    for (i, obj) in enumerate(param_list):
        trailer = [Subscript(Number(str(i)))]
        if isinstance(obj, list):
            map_to_index(obj, trailer, d=d)
        else:
            d[obj] = prefix + trailer
    return d

def tuple_name(param_list):
    if False:
        while True:
            i = 10
    l = []
    for obj in param_list:
        if isinstance(obj, list):
            l.append(tuple_name(obj))
        else:
            l.append(obj)
    return '_'.join(l)