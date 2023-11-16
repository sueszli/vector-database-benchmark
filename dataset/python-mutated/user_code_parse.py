import ast
from typing import List
from .unparse import unparse

class GlobalsVisitor(ast.NodeVisitor):

    def generic_visit(self, node):
        if False:
            while True:
                i = 10
        if isinstance(node, ast.Global):
            raise Exception('No Globals allows')
        ast.NodeVisitor.generic_visit(self, node)

def check_no_returns(module: ast.Module) -> None:
    if False:
        return 10
    for node in module.body:
        if isinstance(node, ast.Return):
            raise Exception('Main body of function cannot return')

def make_return(var_name: str) -> ast.Return:
    if False:
        for i in range(10):
            print('nop')
    name = ast.Name(id=var_name)
    return ast.Return(value=name)

def make_ast_args(args: List[str]) -> ast.arguments:
    if False:
        return 10
    arguments = []
    for arg_name in args:
        arg = ast.arg(arg=arg_name)
        arguments.append(arg)
    return ast.arguments(args=arguments, posonlyargs=[], defaults=[], kwonlyargs=[])

def make_ast_func(name: str, input_kwargs: List[str], output_arg: str, body=List[ast.AST]) -> ast.FunctionDef:
    if False:
        return 10
    args = make_ast_args(input_kwargs)
    r = make_return(output_arg)
    new_body = body + [r]
    f = ast.FunctionDef(name=name, args=args, body=new_body, decorator_list=[], lineno=0)
    return f

def parse_and_wrap_code(func_name: str, raw_code: str, input_kwargs: List[str], output_arg: str) -> str:
    if False:
        i = 10
        return i + 15
    ast_code = ast.parse(raw_code)
    v = GlobalsVisitor()
    v.visit(ast_code)
    check_no_returns(ast_code)
    wrapper_function = make_ast_func(func_name, input_kwargs=input_kwargs, output_arg=output_arg, body=ast_code.body)
    return unparse(wrapper_function)