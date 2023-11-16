from functools import cached_property
from typing import Optional
from vyper import ast as vy_ast

class GlobalContext:

    def __init__(self, module: Optional[vy_ast.Module]=None):
        if False:
            while True:
                i = 10
        self._module = module

    @cached_property
    def functions(self):
        if False:
            i = 10
            return i + 15
        return self._module.get_children(vy_ast.FunctionDef)

    @cached_property
    def variables(self):
        if False:
            while True:
                i = 10
        if self._module is None:
            return None
        variable_decls = self._module.get_children(vy_ast.VariableDecl)
        return {s.target.id: s.target._metadata['varinfo'] for s in variable_decls}

    @property
    def immutables(self):
        if False:
            while True:
                i = 10
        return [t for t in self.variables.values() if t.is_immutable]

    @cached_property
    def immutable_section_bytes(self):
        if False:
            while True:
                i = 10
        return sum([imm.typ.memory_bytes_required for imm in self.immutables])