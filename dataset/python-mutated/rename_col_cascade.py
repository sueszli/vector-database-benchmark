import ast
from visidata import vd, Column, ExprColumn, Sheet
vd.option('rename_cascade', False, 'cascade column renames into expressions')

class Renamer(ast.NodeTransformer):

    def __init__(self, find, replace):
        if False:
            for i in range(10):
                print('nop')
        self.find = find
        self.replace = replace

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.id == self.find:
            node.id = self.replace
        return node

@Column.before
def setName(col, newname):
    if False:
        return 10
    if col.sheet and col.sheet.options.rename_cascade:
        for c in col.sheet.columns:
            if isinstance(c, ExprColumn):
                parsed_expr = ast.parse(c.expr)
                canon_expr = ast.unparse(parsed_expr)
                new_expr = ast.unparse(Renamer(col.name, newname).visit(parsed_expr))
                if new_expr != canon_expr:
                    vd.addUndo(setattr, c, 'expr', c.expr)
                    c.expr = new_expr