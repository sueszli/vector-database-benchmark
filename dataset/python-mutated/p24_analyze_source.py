"""
Topic: 解析与分析Python源码
Desc : 
"""
import ast

class CodeAnalyzer(ast.NodeVisitor):

    def __init__(self):
        if False:
            return 10
        self.loaded = set()
        self.stored = set()
        self.deleted = set()

    def visit_Name(self, node):
        if False:
            i = 10
            return i + 15
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.stored.add(node.id)
        elif isinstance(node.ctx, ast.Del):
            self.deleted.add(node.id)
if __name__ == '__main__':
    code = '\n    for i in range(10):\n        print(i)\n    del i\n    '
    top = ast.parse(code, mode='exec')
    c = CodeAnalyzer()
    c.visit(top)
    print('Loaded:', c.loaded)
    print('Stored:', c.stored)
    print('Deleted:', c.deleted)
import ast
import inspect

class NameLower(ast.NodeVisitor):

    def __init__(self, lowered_names):
        if False:
            return 10
        self.lowered_names = lowered_names

    def visit_FunctionDef(self, node):
        if False:
            print('Hello World!')
        code = '__globals = globals()\n'
        code += '\n'.join(("{0} = __globals['{0}']".format(name) for name in self.lowered_names))
        code_ast = ast.parse(code, mode='exec')
        node.body[:0] = code_ast.body
        self.func = node

def lower_names(*namelist):
    if False:
        i = 10
        return i + 15

    def lower(func):
        if False:
            for i in range(10):
                print('nop')
        srclines = inspect.getsource(func).splitlines()
        for (n, line) in enumerate(srclines):
            if '@lower_names' in line:
                break
        src = '\n'.join(srclines[n + 1:])
        if src.startswith((' ', '\t')):
            src = 'if 1:\n' + src
        top = ast.parse(src, mode='exec')
        cl = NameLower(namelist)
        cl.visit(top)
        temp = {}
        exec(compile(top, '', 'exec'), temp, temp)
        func.__code__ = temp[func.__name__].__code__
        return func
    return lower