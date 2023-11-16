from __future__ import annotations
from typing import Callable
from mypy.plugin import AttributeContext, Plugin
from mypy.types import AnyType, Type, TypeOfAny

class AttrPlugin(Plugin):

    def get_attribute_hook(self, fullname: str) -> Callable[[AttributeContext], Type] | None:
        if False:
            while True:
                i = 10
        if fullname == 'm.Magic.magic_field':
            return magic_field_callback
        if fullname == 'm.Magic.nonexistent_field':
            return nonexistent_field_callback
        return None

def magic_field_callback(ctx: AttributeContext) -> Type:
    if False:
        i = 10
        return i + 15
    return ctx.api.named_generic_type('builtins.str', [])

def nonexistent_field_callback(ctx: AttributeContext) -> Type:
    if False:
        print('Hello World!')
    ctx.api.fail('Field does not exist', ctx.context)
    return AnyType(TypeOfAny.from_error)

def plugin(version: str) -> type[AttrPlugin]:
    if False:
        return 10
    return AttrPlugin