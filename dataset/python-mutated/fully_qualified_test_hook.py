from __future__ import annotations
from typing import Callable
from mypy.plugin import MethodSigContext, Plugin
from mypy.types import CallableType

class FullyQualifiedTestPlugin(Plugin):

    def get_method_signature_hook(self, fullname: str) -> Callable[[MethodSigContext], CallableType] | None:
        if False:
            i = 10
            return i + 15
        if 'FullyQualifiedTest' in fullname:
            assert fullname.startswith('__main__.') and ' of ' not in fullname, fullname
            return my_hook
        return None

def my_hook(ctx: MethodSigContext) -> CallableType:
    if False:
        print('Hello World!')
    return ctx.default_signature.copy_modified(ret_type=ctx.api.named_generic_type('builtins.int', []))

def plugin(version: str) -> type[FullyQualifiedTestPlugin]:
    if False:
        return 10
    return FullyQualifiedTestPlugin