"""Updates all source files to import the same set of __future__ directives.
"""
from lib2to3 import fixer_base
from lib2to3 import pytree
from lib2to3.pgen2 import token
from lib2to3.fixer_util import FromImport, Name, Comma, Newline

def is_docstring(stmt):
    if False:
        while True:
            i = 10
    return isinstance(stmt, pytree.Node) and stmt.children[0].type == token.STRING

class FixFutureImports(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = 'import_from< \'from\' module_name="__future__" \'import\' any >'

    def start_tree(self, tree, filename):
        if False:
            print('Hello World!')
        self.found_future_import = False

    def new_future_import(self, old):
        if False:
            while True:
                i = 10
        new = FromImport('__future__', [Name('absolute_import', prefix=' '), Comma(), Name('division', prefix=' '), Comma(), Name('print_function', prefix=' ')])
        if old is not None:
            new.prefix = old.prefix
        return new

    def transform(self, node, results):
        if False:
            for i in range(10):
                print('nop')
        self.found_future_import = True
        return self.new_future_import(node)

    def finish_tree(self, tree, filename):
        if False:
            return 10
        if self.found_future_import:
            return
        if not isinstance(tree, pytree.Node):
            return
        first_stmt = tree.children[0]
        if is_docstring(first_stmt):
            tree.insert_child(1, Newline())
            pos = 2
        elif first_stmt.prefix:
            newline = Newline()
            newline.prefix = first_stmt.prefix
            first_stmt.prefix = ''
            tree.insert_child(0, newline)
            pos = 1
        else:
            pos = 0
        tree.insert_child(pos, self.new_future_import(None))
        tree.insert_child(pos + 1, Newline())