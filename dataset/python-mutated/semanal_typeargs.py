"""Verify properties of type arguments, like 'int' in C[int] being valid.

This must happen after semantic analysis since there can be placeholder
types until the end of semantic analysis, and these break various type
operations, including subtype checks.
"""
from __future__ import annotations
from typing import Callable
from mypy import errorcodes as codes, message_registry
from mypy.errorcodes import ErrorCode
from mypy.errors import Errors
from mypy.messages import format_type
from mypy.mixedtraverser import MixedTraverserVisitor
from mypy.nodes import ARG_STAR, Block, ClassDef, Context, FakeInfo, FuncItem, MypyFile
from mypy.options import Options
from mypy.scope import Scope
from mypy.subtypes import is_same_type, is_subtype
from mypy.types import AnyType, CallableType, Instance, Parameters, ParamSpecType, TupleType, Type, TypeAliasType, TypeOfAny, TypeVarLikeType, TypeVarTupleType, TypeVarType, UnboundType, UnpackType, flatten_nested_tuples, get_proper_type, get_proper_types, split_with_prefix_and_suffix

class TypeArgumentAnalyzer(MixedTraverserVisitor):

    def __init__(self, errors: Errors, options: Options, is_typeshed_file: bool, named_type: Callable[[str, list[Type]], Instance]) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.errors = errors
        self.options = options
        self.is_typeshed_file = is_typeshed_file
        self.named_type = named_type
        self.scope = Scope()
        self.recurse_into_functions = True
        self.seen_aliases: set[TypeAliasType] = set()

    def visit_mypy_file(self, o: MypyFile) -> None:
        if False:
            print('Hello World!')
        self.errors.set_file(o.path, o.fullname, scope=self.scope, options=self.options)
        with self.scope.module_scope(o.fullname):
            super().visit_mypy_file(o)

    def visit_func(self, defn: FuncItem) -> None:
        if False:
            while True:
                i = 10
        if not self.recurse_into_functions:
            return
        with self.scope.function_scope(defn):
            super().visit_func(defn)

    def visit_class_def(self, defn: ClassDef) -> None:
        if False:
            print('Hello World!')
        with self.scope.class_scope(defn.info):
            super().visit_class_def(defn)

    def visit_block(self, o: Block) -> None:
        if False:
            print('Hello World!')
        if not o.is_unreachable:
            super().visit_block(o)

    def visit_type_alias_type(self, t: TypeAliasType) -> None:
        if False:
            while True:
                i = 10
        super().visit_type_alias_type(t)
        if t in self.seen_aliases:
            return
        self.seen_aliases.add(t)
        assert t.alias is not None, f'Unfixed type alias {t.type_ref}'
        is_error = self.validate_args(t.alias.name, tuple(t.args), t.alias.alias_tvars, t)
        if not is_error:
            get_proper_type(t).accept(self)

    def visit_tuple_type(self, t: TupleType) -> None:
        if False:
            return 10
        t.items = flatten_nested_tuples(t.items)
        super().visit_tuple_type(t)

    def visit_callable_type(self, t: CallableType) -> None:
        if False:
            while True:
                i = 10
        super().visit_callable_type(t)
        if t.is_var_arg:
            star_index = t.arg_kinds.index(ARG_STAR)
            star_type = t.arg_types[star_index]
            if isinstance(star_type, UnpackType):
                p_type = get_proper_type(star_type.type)
                if isinstance(p_type, Instance):
                    assert p_type.type.fullname == 'builtins.tuple'
                    t.arg_types[star_index] = p_type.args[0]

    def visit_instance(self, t: Instance) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_instance(t)
        info = t.type
        if isinstance(info, FakeInfo):
            return
        self.validate_args(info.name, t.args, info.defn.type_vars, t)
        if t.type.fullname == 'builtins.tuple' and len(t.args) == 1:
            arg = t.args[0]
            if isinstance(arg, UnpackType):
                unpacked = get_proper_type(arg.type)
                if isinstance(unpacked, Instance):
                    assert unpacked.type.fullname == 'builtins.tuple'
                    t.args = unpacked.args

    def validate_args(self, name: str, args: tuple[Type, ...], type_vars: list[TypeVarLikeType], ctx: Context) -> bool:
        if False:
            i = 10
            return i + 15
        if any((isinstance(v, TypeVarTupleType) for v in type_vars)):
            prefix = next((i for (i, v) in enumerate(type_vars) if isinstance(v, TypeVarTupleType)))
            tvt = type_vars[prefix]
            assert isinstance(tvt, TypeVarTupleType)
            (start, middle, end) = split_with_prefix_and_suffix(tuple(args), prefix, len(type_vars) - prefix - 1)
            args = start + (TupleType(list(middle), tvt.tuple_fallback),) + end
        is_error = False
        for ((i, arg), tvar) in zip(enumerate(args), type_vars):
            if isinstance(tvar, TypeVarType):
                if isinstance(arg, ParamSpecType):
                    is_error = True
                    self.fail(f'Invalid location for ParamSpec "{arg.name}"', ctx)
                    self.note("You can use ParamSpec as the first argument to Callable, e.g., 'Callable[{}, int]'".format(arg.name), ctx)
                    continue
                if tvar.values:
                    if isinstance(arg, TypeVarType):
                        if self.in_type_alias_expr:
                            continue
                        arg_values = arg.values
                        if not arg_values:
                            is_error = True
                            self.fail(message_registry.INVALID_TYPEVAR_AS_TYPEARG.format(arg.name, name), ctx, code=codes.TYPE_VAR)
                            continue
                    else:
                        arg_values = [arg]
                    if self.check_type_var_values(name, arg_values, tvar.name, tvar.values, ctx):
                        is_error = True
                upper_bound = tvar.upper_bound
                object_upper_bound = type(upper_bound) is Instance and upper_bound.type.fullname == 'builtins.object'
                if not object_upper_bound and (not is_subtype(arg, upper_bound)):
                    if self.in_type_alias_expr and isinstance(arg, TypeVarType):
                        continue
                    is_error = True
                    self.fail(message_registry.INVALID_TYPEVAR_ARG_BOUND.format(format_type(arg, self.options), name, format_type(upper_bound, self.options)), ctx, code=codes.TYPE_VAR)
            elif isinstance(tvar, ParamSpecType):
                if not isinstance(get_proper_type(arg), (ParamSpecType, Parameters, AnyType, UnboundType)):
                    self.fail(f'Can only replace ParamSpec with a parameter types list or another ParamSpec, got {format_type(arg, self.options)}', ctx)
        return is_error

    def visit_unpack_type(self, typ: UnpackType) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().visit_unpack_type(typ)
        proper_type = get_proper_type(typ.type)
        if isinstance(proper_type, TupleType):
            return
        if isinstance(proper_type, TypeVarTupleType):
            return
        if isinstance(proper_type, Instance) and proper_type.type.fullname == 'builtins.tuple':
            return
        if not isinstance(proper_type, (UnboundType, AnyType)):
            self.fail(message_registry.INVALID_UNPACK.format(format_type(proper_type, self.options)), typ.type, code=codes.VALID_TYPE)
        typ.type = self.named_type('builtins.tuple', [AnyType(TypeOfAny.from_error)])

    def check_type_var_values(self, name: str, actuals: list[Type], arg_name: str, valids: list[Type], context: Context) -> bool:
        if False:
            return 10
        is_error = False
        for actual in get_proper_types(actuals):
            if not isinstance(actual, (AnyType, UnboundType)) and (not any((is_same_type(actual, value) for value in valids))):
                is_error = True
                if len(actuals) > 1 or not isinstance(actual, Instance):
                    self.fail(message_registry.INVALID_TYPEVAR_ARG_VALUE.format(name), context, code=codes.TYPE_VAR)
                else:
                    class_name = f'"{name}"'
                    actual_type_name = f'"{actual.type.name}"'
                    self.fail(message_registry.INCOMPATIBLE_TYPEVAR_VALUE.format(arg_name, class_name, actual_type_name), context, code=codes.TYPE_VAR)
        return is_error

    def fail(self, msg: str, context: Context, *, code: ErrorCode | None=None) -> None:
        if False:
            print('Hello World!')
        self.errors.report(context.line, context.column, msg, code=code)

    def note(self, msg: str, context: Context, *, code: ErrorCode | None=None) -> None:
        if False:
            print('Hello World!')
        self.errors.report(context.line, context.column, msg, severity='note', code=code)