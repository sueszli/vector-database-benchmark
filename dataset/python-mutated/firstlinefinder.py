"""
This module provides helper functions to find the first line of a function
body.
"""
import ast

class FindDefFirstLine(ast.NodeVisitor):
    """
    Attributes
    ----------
    first_stmt_line : int or None
        This stores the first statement line number if the definition is found.
        Or, ``None`` if the definition is not found.
    """

    def __init__(self, code):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parameters\n        ----------\n        code :\n            The function's code object.\n        "
        self._co_name = code.co_name
        self._co_firstlineno = code.co_firstlineno
        self.first_stmt_line = None

    def _visit_children(self, node):
        if False:
            i = 10
            return i + 15
        for child in ast.iter_child_nodes(node):
            super().visit(child)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if False:
            return 10
        if node.name == self._co_name:
            possible_start_lines = set([node.lineno])
            if node.decorator_list:
                first_decor = node.decorator_list[0]
                possible_start_lines.add(first_decor.lineno)
            if self._co_firstlineno in possible_start_lines:
                if node.body:
                    first_stmt = node.body[0]
                    if _is_docstring(first_stmt):
                        first_stmt = node.body[1]
                    self.first_stmt_line = first_stmt.lineno
                    return
                else:
                    pass
        self._visit_children(node)

def _is_docstring(node):
    if False:
        return 10
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return True
    return False

def get_func_body_first_lineno(pyfunc):
    if False:
        while True:
            i = 10
    '\n    Look up the first line of function body using the file in\n    ``pyfunc.__code__.co_filename``.\n\n    Returns\n    -------\n    lineno : int; or None\n        The first line number of the function body; or ``None`` if the first\n        line cannot be determined.\n    '
    co = pyfunc.__code__
    try:
        with open(co.co_filename) as fin:
            file_content = fin.read()
    except (FileNotFoundError, OSError):
        return
    else:
        tree = ast.parse(file_content)
        finder = FindDefFirstLine(co)
        finder.visit(tree)
        return finder.first_stmt_line