import ast
import functools
import inspect
import textwrap
from typing import Callable, Generic, Optional, TypeVar, Union, cast
from .utils import inspect_function_scope
T = TypeVar('T')

class CinnLowerLevelIrJit(Generic[T]):

    def __init__(self, fn):
        if False:
            return 10
        self.fn = fn
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find('def'):]
        self.scope = inspect_function_scope(fn)
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.run = self._make_launcher()

    def _make_launcher(self):
        if False:
            i = 10
            return i + 15
        jit_input_args = ', '.join((arg_name for arg_name in self.arg_names))
        lazy_compile = f"\nimport cinn\ndef {self.fn.__name__}({jit_input_args}, target=cinn.common.DefaultHostTarget()):\n    from cinn.compiler import compile\n    jit_inputs = {', '.join([f'{arg}' for arg in self.arg_names])}\n    jit_inputs_signature = {{ i: self._convert_arg_type(arg)                              for i, arg in enumerate(jit_inputs)}}\n    module = compile(self, jit_inputs_signature=jit_inputs_signature, arg_names={self.arg_names}, target=target)\n    module({jit_input_args})\n\n    return module\n        "
        scope = {'self': self}
        exec(lazy_compile, scope)
        return scope[self.fn.__name__]

    def convert_to_llir(self):
        if False:
            return 10
        from cinn.compiler import compile
        return compile(self, just_convert=True)

    def parse(self):
        if False:
            return 10
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        return tree

    def __getitem__(self, target):
        if False:
            return 10
        return cast(T, functools.partial(cast(Callable, self.run), target=target))

    def _convert_arg_type(self, arg):
        if False:
            return 10
        if hasattr(arg, 'dtype'):
            return arg
        elif isinstance(arg, int):
            if -2 ** 21 <= arg and arg <= 2 ** 31 - 1:
                return 'i32'
            elif 2 ** 63 <= arg and arg <= 2 ** 64 - 1:
                return 'u64'
            else:
                return 'i64'
        elif isinstance(arg, float):
            return 'fp32'
        else:
            raise TypeError(f'Unsupported type {type(arg)} for {arg}')

    def __str__(self):
        if False:
            return 10
        return str(self.convert_to_llir())

def to_cinn_llir(fn: Optional[T]=None) -> Union[CinnLowerLevelIrJit[T]]:
    if False:
        while True:
            i = 10

    def decorator(fn: T) -> CinnLowerLevelIrJit[T]:
        if False:
            while True:
                i = 10
        return CinnLowerLevelIrJit(fn)
    if fn is not None:
        return decorator(fn)
    else:
        return decorator