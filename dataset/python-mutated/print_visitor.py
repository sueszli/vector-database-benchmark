"""
Created on Jul 19, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ...asttools import Visitor
import sys
import _ast
from warnings import warn
if sys.version_info.major < 3:
    from StringIO import StringIO
else:
    from io import StringIO

class Indentor(object):

    def __init__(self, printer, indent='    '):
        if False:
            for i in range(10):
                print('nop')
        self.printer = printer
        self.indent = indent

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.printer._indent = self.printer._indent + self.indent

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        indent = self.printer._indent[:-len(self.indent)]
        self.printer._indent = indent
clsname = lambda node: type(node).__name__

def depth(node):
    if False:
        while True:
            i = 10
    return len(flatten(node))

def flatten(node):
    if False:
        while True:
            i = 10
    result = []
    if isinstance(node, _ast.AST):
        for value in ast_values(node):
            result.extend(flatten(value))
    elif isinstance(node, (list, tuple)):
        for child in node:
            result.extend(flatten(child))
    else:
        result.append(node)
    return result

def ast_keys(node):
    if False:
        return 10
    return node._fields

def ast_values(node):
    if False:
        for i in range(10):
            print('nop')
    return [getattr(node, field, None) for field in node._fields]

def ast_items(node):
    if False:
        i = 10
        return i + 15
    return [(field, getattr(node, field, None)) for field in node._fields]

class ASTPrinter(Visitor):

    def __init__(self, indent=' ', level=0, newline='\n'):
        if False:
            for i in range(10):
                print('nop')
        self.out = StringIO()
        self._indent = ''
        self.one_indent = indent
        self.level = level
        self.newline = newline

    def dump(self, file=sys.stdout):
        if False:
            i = 10
            return i + 15
        self.out.seek(0)
        print(self.out.read(), file=file)

    def dumps(self):
        if False:
            return 10
        self.out.seek(0)
        return self.out.read()

    def print(self, text, noindent=False, **kwargs):
        if False:
            while True:
                i = 10
        new_text = text.format(**kwargs)
        print(new_text, file=self.out, sep='', end='')

    def indent(self, level):
        if False:
            while True:
                i = 10
        ident = self.one_indent * level
        return Indentor(self, ident)

    def visitDefault(self, node):
        if False:
            while True:
                i = 10
        nodename = '%s(' % clsname(node)
        self.print(nodename, noindent=True)
        undefined = [attr for attr in node._fields if not hasattr(node, attr)]
        if undefined:
            warn('ast node %r does not have required field(s) %r ' % (clsname(node), undefined), stacklevel=2)
        undefined = [attr for attr in node._attributes if not hasattr(node, attr)]
        if undefined:
            warn('ast does %r not have required attribute(s) %r ' % (clsname(node), undefined), stacklevel=2)
        children = sorted([(attr, getattr(node, attr)) for attr in node._fields if hasattr(node, attr)])
        with self.indent(len(nodename)):
            i = 0
            while children:
                (attr, child) = children.pop(0)
                if isinstance(child, (list, tuple)):
                    text = '{attr}=['.format(attr=attr)
                    self.print(text)
                    with self.indent(len(text)):
                        for (j, inner_child) in enumerate(child):
                            if isinstance(inner_child, _ast.AST):
                                self.visit(inner_child)
                            else:
                                self.print(repr(inner_child))
                            if j < len(child) - 1:
                                self.print(', {nl}{idnt}', nl=self.newline, idnt=self._indent)
                    self.print(']')
                else:
                    text = '{attr}='.format(attr=attr)
                    self.print(text)
                    with self.indent(len(text)):
                        if isinstance(child, _ast.AST):
                            self.visit(child)
                        else:
                            self.print(repr(child))
                if children:
                    self.print(', {nl}{idnt}', nl=self.newline, idnt=self._indent)
                i += 1
        self.print(')')

def dump_ast(ast, indent=' ', newline='\n'):
    if False:
        return 10
    '\n\n    Returns a string representing the ast.\n\n    :param ast: the ast to print.\n    :param indent: how far to indent a newline.\n    :param newline: The newline character.\n\n    '
    visitor = ASTPrinter(indent=indent, level=0, newline=newline)
    visitor.visit(ast)
    return visitor.dumps()

def print_ast(ast, indent=' ', initlevel=0, newline='\n', file=sys.stdout):
    if False:
        for i in range(10):
            print('nop')
    "\n    Pretty print an ast node.\n\n    :param ast: the ast to print.\n    :param indent: how far to indent a newline.\n    :param initlevel: starting indent level\n    :param newline: The newline character.\n    :param file: file object to print to\n\n    To print a short ast you may want to use::\n\n        node = ast.parse(source)\n        print_ast(node, indent='', newline='')\n\n    "
    visitor = ASTPrinter(indent=indent, level=initlevel, newline=newline)
    visitor.visit(ast)
    visitor.dump(file=file)