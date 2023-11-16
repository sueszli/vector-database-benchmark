from __future__ import annotations
from typing import Callable
from mypy.errorcodes import ErrorCode
from mypy.plugin import FunctionContext, Plugin
from mypy.types import AnyType, Type, TypeOfAny
CUSTOM_ERROR = ErrorCode(code='custom', description='', category='Custom')

class CustomErrorCodePlugin(Plugin):

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        if False:
            print('Hello World!')
        if fullname.endswith('.main'):
            return self.emit_error
        return None

    def emit_error(self, ctx: FunctionContext) -> Type:
        if False:
            return 10
        ctx.api.fail('Custom error', ctx.context, code=CUSTOM_ERROR)
        return AnyType(TypeOfAny.from_error)

def plugin(version: str) -> type[CustomErrorCodePlugin]:
    if False:
        for i in range(10):
            print('nop')
    return CustomErrorCodePlugin