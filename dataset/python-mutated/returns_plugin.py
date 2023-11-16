"""
Custom mypy plugin to solve the temporary problem with python typing.

Important: we don't do anything ugly here.
We only solve problems of the current typing implementation.

``mypy`` API docs are here:
https://mypy.readthedocs.io/en/latest/extending_mypy.html

We use ``pytest-mypy-plugins`` to test that it works correctly, see:
https://github.com/mkurnikov/pytest-mypy-plugins
"""
from typing import Callable, ClassVar, Mapping, Optional, Type, final
from mypy.nodes import SymbolTableNode
from mypy.plugin import AttributeContext, FunctionContext, MethodContext, MethodSigContext, Plugin
from mypy.types import CallableType
from mypy.types import Type as MypyType
from returns.contrib.mypy import _consts
from returns.contrib.mypy._features import curry, do_notation, flow, kind, partial, pipe
_FunctionCallback = Callable[[FunctionContext], MypyType]
_FunctionDefCallback = Callable[[Optional[SymbolTableNode]], Callable[[FunctionContext], MypyType]]
_AttributeCallback = Callable[[AttributeContext], MypyType]
_MethodCallback = Callable[[MethodContext], MypyType]
_MethodSigCallback = Callable[[MethodSigContext], CallableType]

@final
class _ReturnsPlugin(Plugin):
    """Our main plugin to dispatch different callbacks to specific features."""
    _function_hook_plugins: ClassVar[Mapping[str, _FunctionCallback]] = {_consts.TYPED_PARTIAL_FUNCTION: partial.analyze, _consts.TYPED_CURRY_FUNCTION: curry.analyze, _consts.TYPED_FLOW_FUNCTION: flow.analyze, _consts.TYPED_PIPE_FUNCTION: pipe.analyze, _consts.TYPED_KIND_DEKIND: kind.dekind}
    _method_sig_hook_plugins: ClassVar[Mapping[str, _MethodSigCallback]] = {_consts.TYPED_PIPE_METHOD: pipe.signature, _consts.TYPED_KIND_KINDED_CALL: kind.kinded_signature}
    _method_hook_plugins: ClassVar[Mapping[str, _MethodCallback]] = {_consts.TYPED_PIPE_METHOD: pipe.infer, _consts.TYPED_KIND_KINDED_CALL: kind.kinded_call, _consts.TYPED_KIND_KINDED_GET: kind.kinded_get_descriptor, **dict.fromkeys(_consts.DO_NOTATION_METHODS, do_notation.analyze)}

    def get_function_hook(self, fullname: str) -> Optional[_FunctionCallback]:
        if False:
            while True:
                i = 10
        '\n        Called for function return types from ``mypy``.\n\n        Runs on each function call in the source code.\n        We are only interested in a particular subset of all functions.\n        So, we return a function handler for them.\n\n        Otherwise, we return ``None``.\n        '
        return self._function_hook_plugins.get(fullname)

    def get_attribute_hook(self, fullname: str) -> Optional[_AttributeCallback]:
        if False:
            while True:
                i = 10
        'Called for any exiting or ``__getattr__`` aatribute access.'
        if fullname.startswith(_consts.TYPED_KINDN_ACCESS):
            return kind.attribute_access
        return None

    def get_method_signature_hook(self, fullname: str) -> Optional[_MethodSigCallback]:
        if False:
            return 10
        'Called for method signature from ``mypy``.'
        return self._method_sig_hook_plugins.get(fullname)

    def get_method_hook(self, fullname: str) -> Optional[_MethodCallback]:
        if False:
            for i in range(10):
                print('nop')
        'Called for method return types from ``mypy``.'
        return self._method_hook_plugins.get(fullname)

def plugin(version: str) -> Type[Plugin]:
    if False:
        while True:
            i = 10
    "Plugin's public API and entrypoint."
    return _ReturnsPlugin