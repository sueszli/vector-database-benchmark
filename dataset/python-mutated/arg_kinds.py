from __future__ import annotations
from typing import Callable
from mypy.plugin import FunctionContext, MethodContext, Plugin
from mypy.types import Type

class ArgKindsPlugin(Plugin):

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        if False:
            i = 10
            return i + 15
        if 'func' in fullname:
            return extract_arg_kinds_from_function
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        if False:
            while True:
                i = 10
        if 'Class.method' in fullname:
            return extract_arg_kinds_from_method
        return None

def extract_arg_kinds_from_function(ctx: FunctionContext) -> Type:
    if False:
        print('Hello World!')
    ctx.api.fail(str([[x.value for x in y] for y in ctx.arg_kinds]), ctx.context)
    return ctx.default_return_type

def extract_arg_kinds_from_method(ctx: MethodContext) -> Type:
    if False:
        return 10
    ctx.api.fail(str([[x.value for x in y] for y in ctx.arg_kinds]), ctx.context)
    return ctx.default_return_type

def plugin(version: str) -> type[ArgKindsPlugin]:
    if False:
        while True:
            i = 10
    return ArgKindsPlugin