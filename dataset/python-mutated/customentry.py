from __future__ import annotations
from typing import Callable
from mypy.plugin import FunctionContext, Plugin
from mypy.types import Type

class MyPlugin(Plugin):

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        if False:
            for i in range(10):
                print('nop')
        if fullname == '__main__.f':
            return my_hook
        assert fullname
        return None

def my_hook(ctx: FunctionContext) -> Type:
    if False:
        return 10
    return ctx.api.named_generic_type('builtins.int', [])

def register(version: str) -> type[MyPlugin]:
    if False:
        return 10
    return MyPlugin