import _ast
import ast

def unparse(tree: _ast.Module) -> str:
    if False:
        i = 10
        return i + 15
    return ast.unparse(tree)