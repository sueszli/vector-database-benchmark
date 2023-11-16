from types import MappingProxyType
from typing import Final, List, Optional, overload
from mypy.checkmember import analyze_member_access
from mypy.nodes import ARG_NAMED, ARG_OPT
from mypy.types import CallableType, FunctionLike
from mypy.types import Type as MypyType
from typing_extensions import Literal
from returns.contrib.mypy._structures.args import FuncArg
from returns.contrib.mypy._structures.types import CallableContext
_KIND_MAPPING: Final = MappingProxyType({ARG_OPT: ARG_NAMED})

@overload
def analyze_call(function: FunctionLike, args: List[FuncArg], ctx: CallableContext, *, show_errors: Literal[True]) -> CallableType:
    if False:
        i = 10
        return i + 15
    'Case when errors are reported and we cannot get ``None``.'

@overload
def analyze_call(function: FunctionLike, args: List[FuncArg], ctx: CallableContext, *, show_errors: bool) -> Optional[CallableType]:
    if False:
        while True:
            i = 10
    'Errors are not reported, we can get ``None`` when errors happen.'

def analyze_call(function, args, ctx, *, show_errors):
    if False:
        for i in range(10):
            print('nop')
    '\n    Analyzes function call based on passed arguments.\n\n    Internally uses ``check_call`` from ``mypy``.\n    It does a lot of magic.\n\n    We also allow to return ``None`` instead of showing errors.\n    This might be helpful for cases when we run intermediate analysis.\n    '
    checker = ctx.api.expr_checker
    with checker.msg.filter_errors(save_filtered_errors=True) as local_errors:
        (return_type, checked_function) = checker.check_call(function, [arg.expression(ctx.context) for arg in args], [_KIND_MAPPING.get(arg.kind, arg.kind) for arg in args], ctx.context, [arg.name for arg in args])
    if not show_errors and local_errors.has_new_errors():
        return None
    checker.msg.add_errors(local_errors.filtered_errors())
    return checked_function

def safe_translate_to_function(function_def: MypyType, ctx: CallableContext) -> MypyType:
    if False:
        print('Hello World!')
    "\n    Transforms many other types to something close to callable type.\n\n    There's why we need it:\n\n    - We can use this on real functions\n    - We can use this on ``@overload`` functions\n    - We can use this on instances with ``__call__``\n    - We can use this on ``Type`` types\n\n    It can probably work with other types as well.\n\n    This function allows us to unify this process.\n    We also need to disable errors, because we explicitly pass empty args.\n\n    This function also resolves all type arguments.\n    "
    checker = ctx.api.expr_checker
    with checker.msg.filter_errors():
        (_return_type, function_def) = checker.check_call(function_def, [], [], ctx.context, [])
    return function_def

def translate_to_function(function_def: MypyType, ctx: CallableContext) -> MypyType:
    if False:
        print('Hello World!')
    "\n    Tryies to translate a type into callable by accessing ``__call__`` attr.\n\n    This might fail with ``mypy`` errors and that's how it must work.\n    This also preserves all type arguments as-is.\n    "
    checker = ctx.api.expr_checker
    return analyze_member_access('__call__', function_def, ctx.context, is_lvalue=False, is_super=False, is_operator=True, msg=checker.msg, original_type=function_def, chk=checker.chk, in_literal_context=checker.is_literal_context())