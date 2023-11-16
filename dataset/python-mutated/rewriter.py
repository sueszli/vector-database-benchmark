import ast
import inspect
import textwrap
import copy
import functools
from types import FunctionType
from typing import cast, Union, Callable, Dict, Optional, Any
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph import Graph
from torch._sources import normalize_source_lines
import torch

class AST_Rewriter(ast.NodeTransformer):
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    def rewrite(self, fn: FunctionType):
        if False:
            print('Hello World!')
        (sourcelines, _) = inspect.getsourcelines(fn)
        sourcelines = normalize_source_lines(sourcelines)
        source = ''.join(sourcelines)
        normalized_str = textwrap.dedent(source)
        source_ast = ast.parse(normalized_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))
        code = compile(dest_ast, '', 'exec')
        globals_dict = copy.copy(fn.__globals__)
        keys_before = set(globals_dict.keys())
        exec(code, globals_dict)
        new_keys = list(set(globals_dict.keys()) - keys_before)
        assert len(new_keys) == 1
        fn_compiled = globals_dict[new_keys[0]]

        def change_func_globals(f, globals):
            if False:
                for i in range(10):
                    print('nop')
            'Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)'
            g = FunctionType(f.__code__, globals, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
            g = functools.update_wrapper(g, f)
            g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
            return g
        return change_func_globals(fn_compiled, globals=fn.__globals__)

    def visit_Assert(self, node):
        if False:
            return 10
        "\n        Swap out the Assert node (Python's `assert`) with a callsite to the\n        symbolically-traceable torch._assert function\n        "
        n = ast.parse('torch._assert()', mode='eval')
        assert isinstance(n, ast.Expression)
        call_node = n.body
        assert isinstance(call_node, ast.Call)
        msg = node.msg if node.msg else ast.Constant(value='', kind=None)
        call_node.args = [node.test, msg]
        expr_wrapper = ast.Expr(value=call_node)
        return ast.copy_location(expr_wrapper, node)

    def visit_AnnAssign(self, node):
        if False:
            i = 10
            return i + 15
        "\n        Swap out Python's AnnAssign with an Assign node where the annotation function is called.\n        Example:\n             Original:\n             y: Tensor_Type(1,2,3, Dyn) = f2(x)\n            Output:\n             y = annotate(f2(x),Tensor_Type((1,2,3,Dyn)))\n        "
        return ast.Assign(targets=[node.target], value=ast.Call(func=ast.Name(id='annotate', ctx=ast.Load()), args=[node.value, node.annotation], keywords=[]))

class RewritingTracer(Tracer):

    def trace(self, root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]]=None) -> Graph:
        if False:
            while True:
                i = 10
        return super().trace(_rewrite(root), concrete_args)

def _rewrite(fn: Union[torch.nn.Module, Callable]) -> Union[torch.nn.Module, Callable]:
    if False:
        return 10
    if isinstance(fn, torch.nn.Module):

        def rewrite_module(m: torch.nn.Module):
            if False:
                i = 10
                return i + 15

            class RewrittenModule(torch.nn.Module):

                def __init__(self, orig):
                    if False:
                        return 10
                    super().__init__()
                    for (k, v) in orig.__dict__.items():
                        if isinstance(v, torch.nn.Module):
                            self.__dict__[k] = copy.copy(rewrite_module(v))
                        else:
                            self.__dict__[k] = copy.copy(v)
            RewrittenModule.forward = AST_Rewriter().rewrite(cast(FunctionType, m.forward))
            return RewrittenModule(m)
        return rewrite_module(fn)
    else:
        return AST_Rewriter().rewrite(cast(FunctionType, fn))