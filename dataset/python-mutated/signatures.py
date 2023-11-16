from dataclasses import dataclass
from typing import List, Optional, Set
import torchgen.api.cpp as aten_cpp
from torchgen.api.types import Binding, CType
from torchgen.model import FunctionSchema, NativeFunction
from .types import contextArg

@dataclass(frozen=True)
class ExecutorchCppSignature:
    """
    This signature is merely a CppSignature with Executorch types (optionally
    contains KernelRuntimeContext as well). The inline definition of
    CppSignature is generated in Functions.h and it's used by unboxing
    functions.
    """
    func: FunctionSchema
    cpp_no_default_args: Set[str]
    prefix: str = ''

    def arguments(self, *, include_context: bool=True) -> List[Binding]:
        if False:
            for i in range(10):
                print('nop')
        return ([contextArg] if include_context else []) + et_cpp.arguments(self.func.arguments, faithful=True, method=False, cpp_no_default_args=self.cpp_no_default_args)

    def name(self) -> str:
        if False:
            while True:
                i = 10
        return self.prefix + aten_cpp.name(self.func, faithful_name_for_out_overloads=True)

    def decl(self, name: Optional[str]=None, *, include_context: bool=True) -> str:
        if False:
            for i in range(10):
                print('nop')
        args_str = ', '.join((a.decl() for a in self.arguments(include_context=include_context)))
        if name is None:
            name = self.name()
        return f'{self.returns_type().cpp_type()} {name}({args_str})'

    def defn(self, name: Optional[str]=None) -> str:
        if False:
            print('Hello World!')
        args = [a.defn() for a in self.arguments()]
        args_str = ', '.join(args)
        if name is None:
            name = self.name()
        return f'{self.returns_type().cpp_type()} {name}({args_str})'

    def returns_type(self) -> CType:
        if False:
            while True:
                i = 10
        return et_cpp.returns_type(self.func.returns)

    @staticmethod
    def from_native_function(f: NativeFunction, *, prefix: str='') -> 'ExecutorchCppSignature':
        if False:
            for i in range(10):
                print('nop')
        return ExecutorchCppSignature(func=f.func, prefix=prefix, cpp_no_default_args=f.cpp_no_default_args)
from torchgen.executorch.api import et_cpp