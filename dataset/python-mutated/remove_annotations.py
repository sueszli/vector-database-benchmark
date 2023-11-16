from __future__ import annotations
import ast
from typing import final, TypeVar
FunctionDefNode = TypeVar('FunctionDefNode', ast.FunctionDef, ast.AsyncFunctionDef)

def remove_annotations(node: ast.AST) -> ast.Module:
    if False:
        for i in range(10):
            print('nop')
    return ast.fix_missing_locations(AnnotationRemover().visit(node))

def _copy_attrs(src: ast.AST, dest: ast.AST) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Copies line and column info from one node to another.\n    '
    dest.lineno = src.lineno
    dest.end_lineno = src.end_lineno
    dest.col_offset = src.col_offset
    dest.end_col_offset = src.end_col_offset

@final
class AnnotationRemover(ast.NodeTransformer):

    def visit_single_arg(self, arg: ast.arg) -> ast.arg:
        if False:
            return 10
        arg.annotation = None
        return arg

    def visit_fn_arguments(self, node: ast.arguments) -> ast.arguments:
        if False:
            print('Hello World!')
        if node.posonlyargs:
            node.posonlyargs = [self.visit_single_arg(a) for a in node.posonlyargs]
        if node.args:
            node.args = [self.visit_single_arg(a) for a in node.args]
        if node.kwonlyargs:
            node.kwonlyargs = [self.visit_single_arg(a) for a in node.kwonlyargs]
        vararg = node.vararg
        if vararg:
            node.vararg = self.visit_single_arg(vararg)
        kwarg = node.kwarg
        if kwarg:
            node.kwarg = self.visit_single_arg(kwarg)
        return node

    def visit_function(self, node: FunctionDefNode) -> FunctionDefNode:
        if False:
            print('Hello World!')
        node.arguments = self.visit_fn_arguments(node.args)
        node.returns = None
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        return node

    def visit_FunctionDef(self, node: FunctionDefNode) -> FunctionDefNode:
        if False:
            i = 10
            return i + 15
        return self.visit_function(node)

    def visit_AsyncFunctionDef(self, node: FunctionDefNode) -> FunctionDefNode:
        if False:
            return 10
        return self.visit_function(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign:
        if False:
            for i in range(10):
                print('nop')
        value = node.value
        if value is None:
            value = ast.Ellipsis()
            value.kind = None
            _copy_attrs(node, value)
        assign = ast.Assign(targets=[node.target], value=value, type_comment=None)
        _copy_attrs(node, assign)
        return assign