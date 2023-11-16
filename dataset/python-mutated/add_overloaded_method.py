from __future__ import annotations
from typing import Callable
from mypy.nodes import ARG_POS, Argument, Var
from mypy.plugin import ClassDefContext, Plugin
from mypy.plugins.common import MethodSpec, add_overloaded_method_to_class

class OverloadedMethodPlugin(Plugin):

    def get_base_class_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        if False:
            return 10
        if 'AddOverloadedMethod' in fullname:
            return add_overloaded_method_hook
        return None

def add_overloaded_method_hook(ctx: ClassDefContext) -> None:
    if False:
        for i in range(10):
            print('nop')
    add_overloaded_method_to_class(ctx.api, ctx.cls, 'method', _generate_method_specs(ctx))
    add_overloaded_method_to_class(ctx.api, ctx.cls, 'clsmethod', _generate_method_specs(ctx), is_classmethod=True)
    add_overloaded_method_to_class(ctx.api, ctx.cls, 'stmethod', _generate_method_specs(ctx), is_staticmethod=True)

def _generate_method_specs(ctx: ClassDefContext) -> list[MethodSpec]:
    if False:
        while True:
            i = 10
    return [MethodSpec(args=[Argument(Var('arg'), ctx.api.named_type('builtins.int'), None, ARG_POS)], return_type=ctx.api.named_type('builtins.str')), MethodSpec(args=[Argument(Var('arg'), ctx.api.named_type('builtins.str'), None, ARG_POS)], return_type=ctx.api.named_type('builtins.int'))]

def plugin(version: str) -> type[OverloadedMethodPlugin]:
    if False:
        i = 10
        return i + 15
    return OverloadedMethodPlugin