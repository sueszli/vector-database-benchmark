from __future__ import annotations
from typing import Callable
from mypy.plugin import ClassDefContext, Plugin

class DummyPlugin(Plugin):

    def get_customize_class_mro_hook(self, fullname: str) -> Callable[[ClassDefContext], None]:
        if False:
            return 10

        def analyze(classdef_ctx: ClassDefContext) -> None:
            if False:
                i = 10
                return i + 15
            pass
        return analyze

def plugin(version: str) -> type[DummyPlugin]:
    if False:
        for i in range(10):
            print('nop')
    return DummyPlugin