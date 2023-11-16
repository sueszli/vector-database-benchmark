"""Semantic analysis of types"""
from __future__ import annotations
import itertools
from contextlib import contextmanager
from typing import Callable, Final, Iterable, Iterator, List, Sequence, Tuple, TypeVar
from typing_extensions import Protocol
from mypy import errorcodes as codes, message_registry, nodes
from mypy.errorcodes import ErrorCode
from mypy.messages import MessageBuilder, format_type_bare, quote_type_string, wrong_type_arg_count
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR, ARG_STAR2, SYMBOL_FUNCBASE_TYPES, ArgKind, Context, Decorator, MypyFile, ParamSpecExpr, PlaceholderNode, SymbolTableNode, TypeAlias, TypeInfo, TypeVarExpr, TypeVarLikeExpr, TypeVarTupleExpr, Var, check_arg_kinds, check_arg_names, get_nongen_builtins
from mypy.options import Options
from mypy.plugin import AnalyzeTypeContext, Plugin, TypeAnalyzerPluginInterface
from mypy.semanal_shared import SemanticAnalyzerCoreInterface, paramspec_args, paramspec_kwargs
from mypy.tvar_scope import TypeVarLikeScope
from mypy.types import ANNOTATED_TYPE_NAMES, ANY_STRATEGY, FINAL_TYPE_NAMES, LITERAL_TYPE_NAMES, NEVER_NAMES, TYPE_ALIAS_NAMES, AnyType, BoolTypeQuery, CallableArgument, CallableType, DeletedType, EllipsisType, ErasedType, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecFlavor, ParamSpecType, PartialType, PlaceholderType, ProperType, RawExpressionType, RequiredType, SyntheticTypeVisitor, TrivialSyntheticTypeTranslator, TupleType, Type, TypeAliasType, TypedDictType, TypeList, TypeOfAny, TypeQuery, TypeType, TypeVarLikeType, TypeVarTupleType, TypeVarType, UnboundType, UninhabitedType, UnionType, UnpackType, callable_with_ellipsis, find_unpack_in_list, flatten_nested_tuples, flatten_nested_unions, get_proper_type, has_type_vars
from mypy.types_utils import is_bad_type_type_item
from mypy.typevars import fill_typevars
T = TypeVar('T')
type_constructors: Final = {'typing.Callable', 'typing.Optional', 'typing.Tuple', 'typing.Type', 'typing.Union', *LITERAL_TYPE_NAMES, *ANNOTATED_TYPE_NAMES}
ARG_KINDS_BY_CONSTRUCTOR: Final = {'mypy_extensions.Arg': ARG_POS, 'mypy_extensions.DefaultArg': ARG_OPT, 'mypy_extensions.NamedArg': ARG_NAMED, 'mypy_extensions.DefaultNamedArg': ARG_NAMED_OPT, 'mypy_extensions.VarArg': ARG_STAR, 'mypy_extensions.KwArg': ARG_STAR2}
GENERIC_STUB_NOT_AT_RUNTIME_TYPES: Final = {'queue.Queue', 'builtins._PathLike', 'asyncio.futures.Future'}
SELF_TYPE_NAMES: Final = {'typing.Self', 'typing_extensions.Self'}

def analyze_type_alias(type: Type, api: SemanticAnalyzerCoreInterface, tvar_scope: TypeVarLikeScope, plugin: Plugin, options: Options, is_typeshed_stub: bool, allow_placeholder: bool=False, in_dynamic_func: bool=False, global_scope: bool=True, allowed_alias_tvars: list[TypeVarLikeType] | None=None) -> tuple[Type, set[str]]:
    if False:
        while True:
            i = 10
    "Analyze r.h.s. of a (potential) type alias definition.\n\n    If `node` is valid as a type alias rvalue, return the resulting type and a set of\n    full names of type aliases it depends on (directly or indirectly).\n    'node' must have been semantically analyzed.\n    "
    analyzer = TypeAnalyser(api, tvar_scope, plugin, options, is_typeshed_stub, defining_alias=True, allow_placeholder=allow_placeholder, prohibit_self_type='type alias target', allowed_alias_tvars=allowed_alias_tvars)
    analyzer.in_dynamic_func = in_dynamic_func
    analyzer.global_scope = global_scope
    res = type.accept(analyzer)
    return (res, analyzer.aliases_used)

def no_subscript_builtin_alias(name: str, propose_alt: bool=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    class_name = name.split('.')[-1]
    msg = f'"{class_name}" is not subscriptable'
    nongen_builtins = get_nongen_builtins((3, 8))
    replacement = nongen_builtins[name]
    if replacement and propose_alt:
        msg += f', use "{replacement}" instead'
    return msg

class TypeAnalyser(SyntheticTypeVisitor[Type], TypeAnalyzerPluginInterface):
    """Semantic analyzer for types.

    Converts unbound types into bound types. This is a no-op for already
    bound types.

    If an incomplete reference is encountered, this does a defer. The
    caller never needs to defer.
    """
    in_dynamic_func: bool = False
    global_scope: bool = True

    def __init__(self, api: SemanticAnalyzerCoreInterface, tvar_scope: TypeVarLikeScope, plugin: Plugin, options: Options, is_typeshed_stub: bool, *, defining_alias: bool=False, allow_tuple_literal: bool=False, allow_unbound_tvars: bool=False, allow_placeholder: bool=False, allow_required: bool=False, allow_param_spec_literals: bool=False, allow_unpack: bool=False, report_invalid_types: bool=True, prohibit_self_type: str | None=None, allowed_alias_tvars: list[TypeVarLikeType] | None=None, allow_type_any: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self.api = api
        self.fail_func = api.fail
        self.note_func = api.note
        self.tvar_scope = tvar_scope
        self.defining_alias = defining_alias
        self.allow_tuple_literal = allow_tuple_literal
        self.nesting_level = 0
        self.always_allow_new_syntax = self.api.is_stub_file or self.api.is_future_flag_set('annotations')
        self.allow_unbound_tvars = allow_unbound_tvars
        if allowed_alias_tvars is None:
            allowed_alias_tvars = []
        self.allowed_alias_tvars = allowed_alias_tvars
        self.allow_placeholder = allow_placeholder
        self.allow_required = allow_required
        self.allow_param_spec_literals = allow_param_spec_literals
        self.allow_ellipsis = False
        self.report_invalid_types = report_invalid_types
        self.plugin = plugin
        self.options = options
        self.is_typeshed_stub = is_typeshed_stub
        self.aliases_used: set[str] = set()
        self.prohibit_self_type = prohibit_self_type
        self.allow_type_any = allow_type_any
        self.allow_type_var_tuple = False
        self.allow_unpack = allow_unpack

    def lookup_qualified(self, name: str, ctx: Context, suppress_errors: bool=False) -> SymbolTableNode | None:
        if False:
            return 10
        return self.api.lookup_qualified(name, ctx, suppress_errors)

    def lookup_fully_qualified(self, name: str) -> SymbolTableNode:
        if False:
            return 10
        return self.api.lookup_fully_qualified(name)

    def visit_unbound_type(self, t: UnboundType, defining_literal: bool=False) -> Type:
        if False:
            i = 10
            return i + 15
        typ = self.visit_unbound_type_nonoptional(t, defining_literal)
        if t.optional:
            return make_optional_type(typ)
        return typ

    def visit_unbound_type_nonoptional(self, t: UnboundType, defining_literal: bool) -> Type:
        if False:
            print('Hello World!')
        sym = self.lookup_qualified(t.name, t)
        if sym is not None:
            node = sym.node
            if isinstance(node, PlaceholderNode):
                if node.becomes_typeinfo:
                    if self.api.final_iteration:
                        self.cannot_resolve_type(t)
                        return AnyType(TypeOfAny.from_error)
                    elif self.allow_placeholder:
                        self.api.defer()
                    else:
                        self.api.record_incomplete_ref()
                    return PlaceholderType(node.fullname, self.anal_array(t.args, allow_param_spec=True, allow_param_spec_literals=True, allow_unpack=True), t.line)
                elif self.api.final_iteration:
                    self.cannot_resolve_type(t)
                    return AnyType(TypeOfAny.from_error)
                else:
                    self.api.record_incomplete_ref()
                    return AnyType(TypeOfAny.special_form)
            if node is None:
                self.fail(f'Internal error (node is None, kind={sym.kind})', t)
                return AnyType(TypeOfAny.special_form)
            fullname = node.fullname
            hook = self.plugin.get_type_analyze_hook(fullname)
            if hook is not None:
                return hook(AnalyzeTypeContext(t, t, self))
            if fullname in get_nongen_builtins(self.options.python_version) and t.args and (not self.always_allow_new_syntax):
                self.fail(no_subscript_builtin_alias(fullname, propose_alt=not self.defining_alias), t)
            tvar_def = self.tvar_scope.get_binding(sym)
            if isinstance(sym.node, ParamSpecExpr):
                if tvar_def is None:
                    if self.allow_unbound_tvars:
                        return t
                    self.fail(f'ParamSpec "{t.name}" is unbound', t, code=codes.VALID_TYPE)
                    return AnyType(TypeOfAny.from_error)
                assert isinstance(tvar_def, ParamSpecType)
                if len(t.args) > 0:
                    self.fail(f'ParamSpec "{t.name}" used with arguments', t, code=codes.VALID_TYPE)
                return ParamSpecType(tvar_def.name, tvar_def.fullname, tvar_def.id, tvar_def.flavor, tvar_def.upper_bound, tvar_def.default, line=t.line, column=t.column)
            if isinstance(sym.node, TypeVarExpr) and self.defining_alias and (not defining_literal) and (tvar_def is None or tvar_def not in self.allowed_alias_tvars):
                self.fail(f'''Can't use bound type variable "{t.name}" to define generic alias''', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            if isinstance(sym.node, TypeVarExpr) and tvar_def is not None:
                assert isinstance(tvar_def, TypeVarType)
                if len(t.args) > 0:
                    self.fail(f'Type variable "{t.name}" used with arguments', t, code=codes.VALID_TYPE)
                return tvar_def.copy_modified(line=t.line, column=t.column)
            if isinstance(sym.node, TypeVarTupleExpr) and (tvar_def is not None and self.defining_alias and (tvar_def not in self.allowed_alias_tvars)):
                self.fail(f'''Can't use bound type variable "{t.name}" to define generic alias''', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            if isinstance(sym.node, TypeVarTupleExpr):
                if tvar_def is None:
                    if self.allow_unbound_tvars:
                        return t
                    self.fail(f'TypeVarTuple "{t.name}" is unbound', t, code=codes.VALID_TYPE)
                    return AnyType(TypeOfAny.from_error)
                assert isinstance(tvar_def, TypeVarTupleType)
                if not self.allow_type_var_tuple:
                    self.fail(f'TypeVarTuple "{t.name}" is only valid with an unpack', t, code=codes.VALID_TYPE)
                    return AnyType(TypeOfAny.from_error)
                if len(t.args) > 0:
                    self.fail(f'Type variable "{t.name}" used with arguments', t, code=codes.VALID_TYPE)
                return TypeVarTupleType(tvar_def.name, tvar_def.fullname, tvar_def.id, tvar_def.upper_bound, sym.node.tuple_fallback, tvar_def.default, line=t.line, column=t.column)
            special = self.try_analyze_special_unbound_type(t, fullname)
            if special is not None:
                return special
            if isinstance(node, TypeAlias):
                self.aliases_used.add(fullname)
                an_args = self.anal_array(t.args, allow_param_spec=True, allow_param_spec_literals=node.has_param_spec_type, allow_unpack=True)
                if node.has_param_spec_type and len(node.alias_tvars) == 1:
                    an_args = self.pack_paramspec_args(an_args)
                disallow_any = self.options.disallow_any_generics and (not self.is_typeshed_stub)
                res = instantiate_type_alias(node, an_args, self.fail, node.no_args, t, self.options, unexpanded_type=t, disallow_any=disallow_any, empty_tuple_index=t.empty_tuple_index)
                if isinstance(res, ProperType) and isinstance(res, Instance) and (not (self.defining_alias and self.nesting_level == 0)) and (not validate_instance(res, self.fail, t.empty_tuple_index)):
                    fix_instance(res, self.fail, self.note, disallow_any=disallow_any, options=self.options, use_generic_error=True, unexpanded_type=t)
                if node.eager:
                    res = get_proper_type(res)
                return res
            elif isinstance(node, TypeInfo):
                return self.analyze_type_with_type_info(node, t.args, t, t.empty_tuple_index)
            elif node.fullname in TYPE_ALIAS_NAMES:
                return AnyType(TypeOfAny.special_form)
            elif node.fullname in ('typing_extensions.Concatenate', 'typing.Concatenate'):
                return self.apply_concatenate_operator(t)
            else:
                return self.analyze_unbound_type_without_type_info(t, sym, defining_literal)
        else:
            return AnyType(TypeOfAny.special_form)

    def pack_paramspec_args(self, an_args: Sequence[Type]) -> list[Type]:
        if False:
            return 10
        count = len(an_args)
        if count == 0:
            return []
        if count == 1 and isinstance(get_proper_type(an_args[0]), AnyType):
            return list(an_args)
        if any((isinstance(a, (Parameters, ParamSpecType)) for a in an_args)):
            if len(an_args) > 1:
                first_wrong = next((arg for arg in an_args if isinstance(arg, (Parameters, ParamSpecType))))
                self.fail('Nested parameter specifications are not allowed', first_wrong, code=codes.VALID_TYPE)
                return [AnyType(TypeOfAny.from_error)]
            return list(an_args)
        first = an_args[0]
        return [Parameters(an_args, [ARG_POS] * count, [None] * count, line=first.line, column=first.column)]

    def cannot_resolve_type(self, t: UnboundType) -> None:
        if False:
            i = 10
            return i + 15
        self.api.fail(f'Cannot resolve name "{t.name}" (possible cyclic definition)', t)
        if self.api.is_func_scope():
            self.note('Recursive types are not allowed at function scope', t)

    def apply_concatenate_operator(self, t: UnboundType) -> Type:
        if False:
            return 10
        if len(t.args) == 0:
            self.api.fail('Concatenate needs type arguments', t, code=codes.VALID_TYPE)
            return AnyType(TypeOfAny.from_error)
        ps = self.anal_type(t.args[-1], allow_param_spec=True, allow_ellipsis=True)
        if not isinstance(ps, (ParamSpecType, Parameters)):
            if isinstance(ps, UnboundType) and self.allow_unbound_tvars:
                sym = self.lookup_qualified(ps.name, t)
                if sym is not None and isinstance(sym.node, ParamSpecExpr):
                    return ps
            self.api.fail('The last parameter to Concatenate needs to be a ParamSpec', t, code=codes.VALID_TYPE)
            return AnyType(TypeOfAny.from_error)
        elif isinstance(ps, ParamSpecType) and ps.prefix.arg_types:
            self.api.fail('Nested Concatenates are invalid', t, code=codes.VALID_TYPE)
        args = self.anal_array(t.args[:-1])
        pre = ps.prefix if isinstance(ps, ParamSpecType) else ps
        names: list[str | None] = [None] * len(args)
        pre = Parameters(args + pre.arg_types, [ARG_POS] * len(args) + pre.arg_kinds, names + pre.arg_names, line=t.line, column=t.column)
        return ps.copy_modified(prefix=pre) if isinstance(ps, ParamSpecType) else pre

    def try_analyze_special_unbound_type(self, t: UnboundType, fullname: str) -> Type | None:
        if False:
            for i in range(10):
                print('nop')
        "Bind special type that is recognized through magic name such as 'typing.Any'.\n\n        Return the bound type if successful, and return None if the type is a normal type.\n        "
        if fullname == 'builtins.None':
            return NoneType()
        elif fullname == 'typing.Any' or fullname == 'builtins.Any':
            return AnyType(TypeOfAny.explicit, line=t.line, column=t.column)
        elif fullname in FINAL_TYPE_NAMES:
            self.fail('Final can be only used as an outermost qualifier in a variable annotation', t, code=codes.VALID_TYPE)
            return AnyType(TypeOfAny.from_error)
        elif fullname == 'typing.Tuple' or (fullname == 'builtins.tuple' and (self.always_allow_new_syntax or self.options.python_version >= (3, 9))):
            sym = self.api.lookup_fully_qualified_or_none('builtins.tuple')
            if not sym or isinstance(sym.node, PlaceholderNode):
                if self.api.is_incomplete_namespace('builtins'):
                    self.api.record_incomplete_ref()
                else:
                    self.fail('Name "tuple" is not defined', t)
                return AnyType(TypeOfAny.special_form)
            if len(t.args) == 0 and (not t.empty_tuple_index):
                any_type = self.get_omitted_any(t)
                return self.named_type('builtins.tuple', [any_type], line=t.line, column=t.column)
            if len(t.args) == 2 and isinstance(t.args[1], EllipsisType):
                instance = self.named_type('builtins.tuple', [self.anal_type(t.args[0])])
                instance.line = t.line
                return instance
            return self.tuple_type(self.anal_array(t.args, allow_unpack=True), line=t.line, column=t.column)
        elif fullname == 'typing.Union':
            items = self.anal_array(t.args)
            return UnionType.make_union(items)
        elif fullname == 'typing.Optional':
            if len(t.args) != 1:
                self.fail('Optional[...] must have exactly one type argument', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            item = self.anal_type(t.args[0])
            return make_optional_type(item)
        elif fullname == 'typing.Callable':
            return self.analyze_callable_type(t)
        elif fullname == 'typing.Type' or (fullname == 'builtins.type' and (self.always_allow_new_syntax or self.options.python_version >= (3, 9))):
            if len(t.args) == 0:
                if fullname == 'typing.Type':
                    any_type = self.get_omitted_any(t)
                    return TypeType(any_type, line=t.line, column=t.column)
                else:
                    return None
            if len(t.args) != 1:
                type_str = 'Type[...]' if fullname == 'typing.Type' else 'type[...]'
                self.fail(type_str + ' must have exactly one type argument', t, code=codes.VALID_TYPE)
            item = self.anal_type(t.args[0])
            if is_bad_type_type_item(item):
                self.fail("Type[...] can't contain another Type[...]", t, code=codes.VALID_TYPE)
                item = AnyType(TypeOfAny.from_error)
            return TypeType.make_normalized(item, line=t.line, column=t.column)
        elif fullname == 'typing.ClassVar':
            if self.nesting_level > 0:
                self.fail('Invalid type: ClassVar nested inside other type', t, code=codes.VALID_TYPE)
            if len(t.args) == 0:
                return AnyType(TypeOfAny.from_omitted_generics, line=t.line, column=t.column)
            if len(t.args) != 1:
                self.fail('ClassVar[...] must have at most one type argument', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            return self.anal_type(t.args[0])
        elif fullname in NEVER_NAMES:
            return UninhabitedType(is_noreturn=True)
        elif fullname in LITERAL_TYPE_NAMES:
            return self.analyze_literal_type(t)
        elif fullname in ANNOTATED_TYPE_NAMES:
            if len(t.args) < 2:
                self.fail('Annotated[...] must have exactly one type argument and at least one annotation', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            return self.anal_type(t.args[0])
        elif fullname in ('typing_extensions.Required', 'typing.Required'):
            if not self.allow_required:
                self.fail('Required[] can be only used in a TypedDict definition', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            if len(t.args) != 1:
                self.fail('Required[] must have exactly one type argument', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            return RequiredType(self.anal_type(t.args[0]), required=True)
        elif fullname in ('typing_extensions.NotRequired', 'typing.NotRequired'):
            if not self.allow_required:
                self.fail('NotRequired[] can be only used in a TypedDict definition', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            if len(t.args) != 1:
                self.fail('NotRequired[] must have exactly one type argument', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            return RequiredType(self.anal_type(t.args[0]), required=False)
        elif self.anal_type_guard_arg(t, fullname) is not None:
            return self.named_type('builtins.bool')
        elif fullname in ('typing.Unpack', 'typing_extensions.Unpack'):
            if len(t.args) != 1:
                self.fail('Unpack[...] requires exactly one type argument', t)
                return AnyType(TypeOfAny.from_error)
            if not self.allow_unpack:
                self.fail(message_registry.INVALID_UNPACK_POSITION, t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            self.allow_type_var_tuple = True
            result = UnpackType(self.anal_type(t.args[0]), line=t.line, column=t.column)
            self.allow_type_var_tuple = False
            return result
        elif fullname in SELF_TYPE_NAMES:
            if t.args:
                self.fail('Self type cannot have type arguments', t)
            if self.prohibit_self_type is not None:
                self.fail(f'Self type cannot be used in {self.prohibit_self_type}', t)
                return AnyType(TypeOfAny.from_error)
            if self.api.type is None:
                self.fail('Self type is only allowed in annotations within class definition', t)
                return AnyType(TypeOfAny.from_error)
            if self.api.type.has_base('builtins.type'):
                self.fail('Self type cannot be used in a metaclass', t)
            if self.api.type.self_type is not None:
                if self.api.type.is_final:
                    return fill_typevars(self.api.type)
                return self.api.type.self_type.copy_modified(line=t.line, column=t.column)
            self.fail('Unexpected Self type', t)
            return AnyType(TypeOfAny.from_error)
        return None

    def get_omitted_any(self, typ: Type, fullname: str | None=None) -> AnyType:
        if False:
            while True:
                i = 10
        disallow_any = not self.is_typeshed_stub and self.options.disallow_any_generics
        return get_omitted_any(disallow_any, self.fail, self.note, typ, self.options, fullname)

    def analyze_type_with_type_info(self, info: TypeInfo, args: Sequence[Type], ctx: Context, empty_tuple_index: bool) -> Type:
        if False:
            return 10
        "Bind unbound type when were able to find target TypeInfo.\n\n        This handles simple cases like 'int', 'modname.UserClass[str]', etc.\n        "
        if len(args) > 0 and info.fullname == 'builtins.tuple':
            fallback = Instance(info, [AnyType(TypeOfAny.special_form)], ctx.line)
            return TupleType(self.anal_array(args, allow_unpack=True), fallback, ctx.line)
        instance = Instance(info, self.anal_array(args, allow_param_spec=True, allow_param_spec_literals=info.has_param_spec_type, allow_unpack=True), ctx.line, ctx.column)
        if len(info.type_vars) == 1 and info.has_param_spec_type:
            instance.args = tuple(self.pack_paramspec_args(instance.args))
        instance.args = tuple(flatten_nested_tuples(instance.args))
        if not (self.defining_alias and self.nesting_level == 0) and (not validate_instance(instance, self.fail, empty_tuple_index)):
            fix_instance(instance, self.fail, self.note, disallow_any=self.options.disallow_any_generics and (not self.is_typeshed_stub), options=self.options)
        tup = info.tuple_type
        if tup is not None:
            if info.special_alias:
                return instantiate_type_alias(info.special_alias, self.anal_array(args, allow_unpack=True), self.fail, False, ctx, self.options, use_standard_error=True)
            return tup.copy_modified(items=self.anal_array(tup.items, allow_unpack=True), fallback=instance)
        td = info.typeddict_type
        if td is not None:
            if info.special_alias:
                return instantiate_type_alias(info.special_alias, self.anal_array(args, allow_unpack=True), self.fail, False, ctx, self.options, use_standard_error=True)
            return td.copy_modified(item_types=self.anal_array(list(td.items.values())), fallback=instance)
        if info.fullname == 'types.NoneType':
            self.fail('NoneType should not be used as a type, please use None instead', ctx, code=codes.VALID_TYPE)
            return NoneType(ctx.line, ctx.column)
        return instance

    def analyze_unbound_type_without_type_info(self, t: UnboundType, sym: SymbolTableNode, defining_literal: bool) -> Type:
        if False:
            while True:
                i = 10
        "Figure out what an unbound type that doesn't refer to a TypeInfo node means.\n\n        This is something unusual. We try our best to find out what it is.\n        "
        name = sym.fullname
        if name is None:
            assert sym.node is not None
            name = sym.node.name
        if isinstance(sym.node, Var):
            typ = get_proper_type(sym.node.type)
            if isinstance(typ, AnyType):
                return AnyType(TypeOfAny.from_unimported_type, missing_import_name=typ.missing_import_name)
            elif self.allow_type_any:
                if isinstance(typ, Instance) and typ.type.fullname == 'builtins.type':
                    return AnyType(TypeOfAny.special_form)
                if isinstance(typ, TypeType) and isinstance(typ.item, AnyType):
                    return AnyType(TypeOfAny.from_another_any, source_any=typ.item)
        unbound_tvar = isinstance(sym.node, (TypeVarExpr, TypeVarTupleExpr)) and self.tvar_scope.get_binding(sym) is None
        if self.allow_unbound_tvars and unbound_tvar:
            return t
        if isinstance(sym.node, Var) and sym.node.info and sym.node.info.is_enum:
            value = sym.node.name
            base_enum_short_name = sym.node.info.name
            if not defining_literal:
                msg = message_registry.INVALID_TYPE_RAW_ENUM_VALUE.format(base_enum_short_name, value)
                self.fail(msg.value, t, code=msg.code)
                return AnyType(TypeOfAny.from_error)
            return LiteralType(value=value, fallback=Instance(sym.node.info, [], line=t.line, column=t.column), line=t.line, column=t.column)
        t = t.copy_modified(args=self.anal_array(t.args))
        notes: list[str] = []
        if isinstance(sym.node, Var):
            notes.append('See https://mypy.readthedocs.io/en/stable/common_issues.html#variables-vs-type-aliases')
            message = 'Variable "{}" is not valid as a type'
        elif isinstance(sym.node, (SYMBOL_FUNCBASE_TYPES, Decorator)):
            message = 'Function "{}" is not valid as a type'
            if name == 'builtins.any':
                notes.append('Perhaps you meant "typing.Any" instead of "any"?')
            elif name == 'builtins.callable':
                notes.append('Perhaps you meant "typing.Callable" instead of "callable"?')
            else:
                notes.append('Perhaps you need "Callable[...]" or a callback protocol?')
        elif isinstance(sym.node, MypyFile):
            message = 'Module "{}" is not valid as a type'
            notes.append('Perhaps you meant to use a protocol matching the module structure?')
        elif unbound_tvar:
            message = 'Type variable "{}" is unbound'
            short = name.split('.')[-1]
            notes.append('(Hint: Use "Generic[{}]" or "Protocol[{}]" base class to bind "{}" inside a class)'.format(short, short, short))
            notes.append('(Hint: Use "{}" in function signature to bind "{}" inside a function)'.format(short, short))
        else:
            message = 'Cannot interpret reference "{}" as a type'
        if not defining_literal:
            self.fail(message.format(name), t, code=codes.VALID_TYPE)
            for note in notes:
                self.note(note, t, code=codes.VALID_TYPE)
        return t

    def visit_any(self, t: AnyType) -> Type:
        if False:
            return 10
        return t

    def visit_none_type(self, t: NoneType) -> Type:
        if False:
            return 10
        return t

    def visit_uninhabited_type(self, t: UninhabitedType) -> Type:
        if False:
            return 10
        return t

    def visit_erased_type(self, t: ErasedType) -> Type:
        if False:
            print('Hello World!')
        assert False, 'Internal error: Unexpected erased type'

    def visit_deleted_type(self, t: DeletedType) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_type_list(self, t: TypeList) -> Type:
        if False:
            while True:
                i = 10
        if self.allow_param_spec_literals:
            params = self.analyze_callable_args(t)
            if params:
                (ts, kinds, names) = params
                return Parameters(self.anal_array(ts), kinds, names, line=t.line, column=t.column)
            else:
                return AnyType(TypeOfAny.from_error)
        else:
            self.fail('Bracketed expression "[...]" is not valid as a type', t, code=codes.VALID_TYPE)
            if len(t.items) == 1:
                self.note('Did you mean "List[...]"?', t)
            return AnyType(TypeOfAny.from_error)

    def visit_callable_argument(self, t: CallableArgument) -> Type:
        if False:
            return 10
        self.fail('Invalid type', t, code=codes.VALID_TYPE)
        return AnyType(TypeOfAny.from_error)

    def visit_instance(self, t: Instance) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_type_alias_type(self, t: TypeAliasType) -> Type:
        if False:
            print('Hello World!')
        return t

    def visit_type_var(self, t: TypeVarType) -> Type:
        if False:
            i = 10
            return i + 15
        return t

    def visit_param_spec(self, t: ParamSpecType) -> Type:
        if False:
            print('Hello World!')
        return t

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_unpack_type(self, t: UnpackType) -> Type:
        if False:
            while True:
                i = 10
        if not self.allow_unpack:
            self.fail(message_registry.INVALID_UNPACK_POSITION, t.type, code=codes.VALID_TYPE)
            return AnyType(TypeOfAny.from_error)
        self.allow_type_var_tuple = True
        result = UnpackType(self.anal_type(t.type), from_star_syntax=t.from_star_syntax)
        self.allow_type_var_tuple = False
        return result

    def visit_parameters(self, t: Parameters) -> Type:
        if False:
            while True:
                i = 10
        raise NotImplementedError('ParamSpec literals cannot have unbound TypeVars')

    def visit_callable_type(self, t: CallableType, nested: bool=True) -> Type:
        if False:
            i = 10
            return i + 15
        with self.tvar_scope_frame():
            unpacked_kwargs = False
            if self.defining_alias:
                variables = t.variables
            else:
                (variables, _) = self.bind_function_type_variables(t, t)
            special = self.anal_type_guard(t.ret_type)
            arg_kinds = t.arg_kinds
            if len(arg_kinds) >= 2 and arg_kinds[-2] == ARG_STAR and (arg_kinds[-1] == ARG_STAR2):
                arg_types = self.anal_array(t.arg_types[:-2], nested=nested) + [self.anal_star_arg_type(t.arg_types[-2], ARG_STAR, nested=nested), self.anal_star_arg_type(t.arg_types[-1], ARG_STAR2, nested=nested)]
                if nested and isinstance(arg_types[-1], UnpackType):
                    unpacked = get_proper_type(arg_types[-1].type)
                    if isinstance(unpacked, TypedDictType):
                        arg_types[-1] = unpacked
                        unpacked_kwargs = True
                    arg_types = self.check_unpacks_in_list(arg_types)
            else:
                star_index = None
                if ARG_STAR in arg_kinds:
                    star_index = arg_kinds.index(ARG_STAR)
                star2_index = None
                if ARG_STAR2 in arg_kinds:
                    star2_index = arg_kinds.index(ARG_STAR2)
                arg_types = []
                for (i, ut) in enumerate(t.arg_types):
                    at = self.anal_type(ut, nested=nested, allow_unpack=i in (star_index, star2_index))
                    if nested and isinstance(at, UnpackType) and (i == star_index):
                        p_at = get_proper_type(at.type)
                        if isinstance(p_at, TypedDictType) and (not at.from_star_syntax):
                            at = p_at
                            arg_kinds[i] = ARG_STAR2
                            unpacked_kwargs = True
                    arg_types.append(at)
                if nested:
                    arg_types = self.check_unpacks_in_list(arg_types)
            arg_kinds = t.arg_kinds[:len(arg_types)]
            arg_names = t.arg_names[:len(arg_types)]
            ret = t.copy_modified(arg_types=arg_types, arg_kinds=arg_kinds, arg_names=arg_names, ret_type=self.anal_type(t.ret_type, nested=nested), fallback=t.fallback if t.fallback.type else self.named_type('builtins.function'), variables=self.anal_var_defs(variables), type_guard=special, unpack_kwargs=unpacked_kwargs)
        return ret

    def anal_type_guard(self, t: Type) -> Type | None:
        if False:
            i = 10
            return i + 15
        if isinstance(t, UnboundType):
            sym = self.lookup_qualified(t.name, t)
            if sym is not None and sym.node is not None:
                return self.anal_type_guard_arg(t, sym.node.fullname)
        return None

    def anal_type_guard_arg(self, t: UnboundType, fullname: str) -> Type | None:
        if False:
            return 10
        if fullname in ('typing_extensions.TypeGuard', 'typing.TypeGuard'):
            if len(t.args) != 1:
                self.fail('TypeGuard must have exactly one type argument', t, code=codes.VALID_TYPE)
                return AnyType(TypeOfAny.from_error)
            return self.anal_type(t.args[0])
        return None

    def anal_star_arg_type(self, t: Type, kind: ArgKind, nested: bool) -> Type:
        if False:
            i = 10
            return i + 15
        'Analyze signature argument type for *args and **kwargs argument.'
        if isinstance(t, UnboundType) and t.name and ('.' in t.name) and (not t.args):
            components = t.name.split('.')
            tvar_name = '.'.join(components[:-1])
            sym = self.lookup_qualified(tvar_name, t)
            if sym is not None and isinstance(sym.node, ParamSpecExpr):
                tvar_def = self.tvar_scope.get_binding(sym)
                if isinstance(tvar_def, ParamSpecType):
                    if kind == ARG_STAR:
                        make_paramspec = paramspec_args
                        if components[-1] != 'args':
                            self.fail(f'Use "{tvar_name}.args" for variadic "*" parameter', t, code=codes.VALID_TYPE)
                    elif kind == ARG_STAR2:
                        make_paramspec = paramspec_kwargs
                        if components[-1] != 'kwargs':
                            self.fail(f'Use "{tvar_name}.kwargs" for variadic "**" parameter', t, code=codes.VALID_TYPE)
                    else:
                        assert False, kind
                    return make_paramspec(tvar_def.name, tvar_def.fullname, tvar_def.id, named_type_func=self.named_type, line=t.line, column=t.column)
        return self.anal_type(t, nested=nested, allow_unpack=True)

    def visit_overloaded(self, t: Overloaded) -> Type:
        if False:
            return 10
        return t

    def visit_tuple_type(self, t: TupleType) -> Type:
        if False:
            i = 10
            return i + 15
        if t.implicit and (not self.allow_tuple_literal):
            self.fail('Syntax error in type annotation', t, code=codes.SYNTAX)
            if len(t.items) == 0:
                self.note('Suggestion: Use Tuple[()] instead of () for an empty tuple, or None for a function without a return value', t, code=codes.SYNTAX)
            elif len(t.items) == 1:
                self.note('Suggestion: Is there a spurious trailing comma?', t, code=codes.SYNTAX)
            else:
                self.note('Suggestion: Use Tuple[T1, ..., Tn] instead of (T1, ..., Tn)', t, code=codes.SYNTAX)
            return AnyType(TypeOfAny.from_error)
        any_type = AnyType(TypeOfAny.special_form)
        fallback = t.partial_fallback if t.partial_fallback.type else self.named_type('builtins.tuple', [any_type])
        return TupleType(self.anal_array(t.items, allow_unpack=True), fallback, t.line)

    def visit_typeddict_type(self, t: TypedDictType) -> Type:
        if False:
            return 10
        items = {item_name: self.anal_type(item_type) for (item_name, item_type) in t.items.items()}
        return TypedDictType(items, set(t.required_keys), t.fallback)

    def visit_raw_expression_type(self, t: RawExpressionType) -> Type:
        if False:
            while True:
                i = 10
        if self.report_invalid_types:
            if t.base_type_name in ('builtins.int', 'builtins.bool'):
                msg = f'Invalid type: try using Literal[{repr(t.literal_value)}] instead?'
            elif t.base_type_name in ('builtins.float', 'builtins.complex'):
                msg = f'Invalid type: {t.simple_name()} literals cannot be used as a type'
            else:
                msg = 'Invalid type comment or annotation'
            self.fail(msg, t, code=codes.VALID_TYPE)
            if t.note is not None:
                self.note(t.note, t, code=codes.VALID_TYPE)
        return AnyType(TypeOfAny.from_error, line=t.line, column=t.column)

    def visit_literal_type(self, t: LiteralType) -> Type:
        if False:
            print('Hello World!')
        return t

    def visit_union_type(self, t: UnionType) -> Type:
        if False:
            while True:
                i = 10
        if t.uses_pep604_syntax is True and t.is_evaluated is True and (not self.always_allow_new_syntax) and (not self.options.python_version >= (3, 10)):
            self.fail('X | Y syntax for unions requires Python 3.10', t, code=codes.SYNTAX)
        return UnionType(self.anal_array(t.items), t.line)

    def visit_partial_type(self, t: PartialType) -> Type:
        if False:
            while True:
                i = 10
        assert False, 'Internal error: Unexpected partial type'

    def visit_ellipsis_type(self, t: EllipsisType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        if self.allow_ellipsis or self.allow_param_spec_literals:
            any_type = AnyType(TypeOfAny.explicit)
            return Parameters([any_type, any_type], [ARG_STAR, ARG_STAR2], [None, None], is_ellipsis_args=True)
        else:
            self.fail('Unexpected "..."', t)
            return AnyType(TypeOfAny.from_error)

    def visit_type_type(self, t: TypeType) -> Type:
        if False:
            return 10
        return TypeType.make_normalized(self.anal_type(t.item), line=t.line)

    def visit_placeholder_type(self, t: PlaceholderType) -> Type:
        if False:
            while True:
                i = 10
        n = None if not t.fullname or '.' not in t.fullname else self.api.lookup_fully_qualified(t.fullname)
        if not n or isinstance(n.node, PlaceholderNode):
            self.api.defer()
            return t
        else:
            assert isinstance(n.node, TypeInfo)
            return self.analyze_type_with_type_info(n.node, t.args, t, False)

    def analyze_callable_args_for_paramspec(self, callable_args: Type, ret_type: Type, fallback: Instance) -> CallableType | None:
        if False:
            return 10
        "Construct a 'Callable[P, RET]', where P is ParamSpec, return None if we cannot."
        if not isinstance(callable_args, UnboundType):
            return None
        sym = self.lookup_qualified(callable_args.name, callable_args)
        if sym is None:
            return None
        tvar_def = self.tvar_scope.get_binding(sym)
        if not isinstance(tvar_def, ParamSpecType):
            if tvar_def is None and self.allow_unbound_tvars and isinstance(sym.node, ParamSpecExpr):
                return callable_with_ellipsis(AnyType(TypeOfAny.explicit), ret_type=ret_type, fallback=fallback)
            return None
        return CallableType([paramspec_args(tvar_def.name, tvar_def.fullname, tvar_def.id, named_type_func=self.named_type), paramspec_kwargs(tvar_def.name, tvar_def.fullname, tvar_def.id, named_type_func=self.named_type)], [nodes.ARG_STAR, nodes.ARG_STAR2], [None, None], ret_type=ret_type, fallback=fallback)

    def analyze_callable_args_for_concatenate(self, callable_args: Type, ret_type: Type, fallback: Instance) -> CallableType | AnyType | None:
        if False:
            while True:
                i = 10
        "Construct a 'Callable[C, RET]', where C is Concatenate[..., P], returning None if we\n        cannot.\n        "
        if not isinstance(callable_args, UnboundType):
            return None
        sym = self.lookup_qualified(callable_args.name, callable_args)
        if sym is None:
            return None
        if sym.node is None:
            return None
        if sym.node.fullname not in ('typing_extensions.Concatenate', 'typing.Concatenate'):
            return None
        tvar_def = self.anal_type(callable_args, allow_param_spec=True)
        if not isinstance(tvar_def, (ParamSpecType, Parameters)):
            if self.allow_unbound_tvars and isinstance(tvar_def, UnboundType):
                sym = self.lookup_qualified(tvar_def.name, callable_args)
                if sym is not None and isinstance(sym.node, ParamSpecExpr):
                    return callable_with_ellipsis(AnyType(TypeOfAny.explicit), ret_type=ret_type, fallback=fallback)
            return AnyType(TypeOfAny.from_error)
        if isinstance(tvar_def, Parameters):
            return CallableType(arg_types=tvar_def.arg_types, arg_names=tvar_def.arg_names, arg_kinds=tvar_def.arg_kinds, ret_type=ret_type, fallback=fallback, from_concatenate=True)
        prefix = tvar_def.prefix
        return CallableType([*prefix.arg_types, paramspec_args(tvar_def.name, tvar_def.fullname, tvar_def.id, named_type_func=self.named_type), paramspec_kwargs(tvar_def.name, tvar_def.fullname, tvar_def.id, named_type_func=self.named_type)], [*prefix.arg_kinds, nodes.ARG_STAR, nodes.ARG_STAR2], [*prefix.arg_names, None, None], ret_type=ret_type, fallback=fallback, from_concatenate=True)

    def analyze_callable_type(self, t: UnboundType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        fallback = self.named_type('builtins.function')
        if len(t.args) == 0:
            any_type = self.get_omitted_any(t)
            ret = callable_with_ellipsis(any_type, any_type, fallback)
        elif len(t.args) == 2:
            callable_args = t.args[0]
            ret_type = t.args[1]
            if isinstance(callable_args, TypeList):
                analyzed_args = self.analyze_callable_args(callable_args)
                if analyzed_args is None:
                    return AnyType(TypeOfAny.from_error)
                (args, kinds, names) = analyzed_args
                ret = CallableType(args, kinds, names, ret_type=ret_type, fallback=fallback)
            elif isinstance(callable_args, EllipsisType):
                ret = callable_with_ellipsis(AnyType(TypeOfAny.explicit), ret_type=ret_type, fallback=fallback)
            else:
                with self.tvar_scope_frame():
                    variables = []
                    for (name, tvar_expr) in self.find_type_var_likes(callable_args):
                        variables.append(self.tvar_scope.bind_new(name, tvar_expr))
                    maybe_ret = self.analyze_callable_args_for_paramspec(callable_args, ret_type, fallback) or self.analyze_callable_args_for_concatenate(callable_args, ret_type, fallback)
                    if isinstance(maybe_ret, CallableType):
                        maybe_ret = maybe_ret.copy_modified(variables=variables)
                if maybe_ret is None:
                    self.fail('The first argument to Callable must be a list of types, parameter specification, or "..."', t, code=codes.VALID_TYPE)
                    self.note('See https://mypy.readthedocs.io/en/stable/kinds_of_types.html#callable-types-and-lambdas', t)
                    return AnyType(TypeOfAny.from_error)
                elif isinstance(maybe_ret, AnyType):
                    return maybe_ret
                ret = maybe_ret
        else:
            if self.options.disallow_any_generics:
                self.fail('Please use "Callable[[<parameters>], <return type>]"', t)
            else:
                self.fail('Please use "Callable[[<parameters>], <return type>]" or "Callable"', t)
            return AnyType(TypeOfAny.from_error)
        assert isinstance(ret, CallableType)
        return ret.accept(self)

    def refers_to_full_names(self, arg: UnboundType, names: Sequence[str]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        sym = self.lookup_qualified(arg.name, arg)
        if sym is not None:
            if sym.fullname in names:
                return True
        return False

    def analyze_callable_args(self, arglist: TypeList) -> tuple[list[Type], list[ArgKind], list[str | None]] | None:
        if False:
            while True:
                i = 10
        args: list[Type] = []
        kinds: list[ArgKind] = []
        names: list[str | None] = []
        seen_unpack = False
        unpack_types: list[Type] = []
        invalid_unpacks: list[Type] = []
        second_unpack_last = False
        for (i, arg) in enumerate(arglist.items):
            if isinstance(arg, CallableArgument):
                args.append(arg.typ)
                names.append(arg.name)
                if arg.constructor is None:
                    return None
                found = self.lookup_qualified(arg.constructor, arg)
                if found is None:
                    return None
                elif found.fullname not in ARG_KINDS_BY_CONSTRUCTOR:
                    self.fail(f'Invalid argument constructor "{found.fullname}"', arg)
                    return None
                else:
                    assert found.fullname is not None
                    kind = ARG_KINDS_BY_CONSTRUCTOR[found.fullname]
                    kinds.append(kind)
                    if arg.name is not None and kind.is_star():
                        self.fail(f'{arg.constructor} arguments should not have names', arg)
                        return None
            elif isinstance(arg, UnboundType) and self.refers_to_full_names(arg, ('typing_extensions.Unpack', 'typing.Unpack')) or isinstance(arg, UnpackType):
                if seen_unpack:
                    if i == len(arglist.items) - 1 and (not invalid_unpacks):
                        second_unpack_last = True
                    invalid_unpacks.append(arg)
                    continue
                seen_unpack = True
                unpack_types.append(arg)
            elif seen_unpack:
                unpack_types.append(arg)
            else:
                args.append(arg)
                kinds.append(ARG_POS)
                names.append(None)
        if seen_unpack:
            if len(unpack_types) == 1:
                args.append(unpack_types[0])
            else:
                first = unpack_types[0]
                if isinstance(first, UnpackType):
                    first = first.type
                args.append(UnpackType(self.tuple_type(unpack_types, line=first.line, column=first.column)))
            kinds.append(ARG_STAR)
            names.append(None)
        for arg in invalid_unpacks:
            args.append(arg)
            kinds.append(ARG_STAR2 if second_unpack_last else ARG_STAR)
            names.append(None)
        check_arg_names(names, [arglist] * len(args), self.fail, 'Callable')
        check_arg_kinds(kinds, [arglist] * len(args), self.fail)
        return (args, kinds, names)

    def analyze_literal_type(self, t: UnboundType) -> Type:
        if False:
            return 10
        if len(t.args) == 0:
            self.fail('Literal[...] must have at least one parameter', t, code=codes.VALID_TYPE)
            return AnyType(TypeOfAny.from_error)
        output: list[Type] = []
        for (i, arg) in enumerate(t.args):
            analyzed_types = self.analyze_literal_param(i + 1, arg, t)
            if analyzed_types is None:
                return AnyType(TypeOfAny.from_error)
            else:
                output.extend(analyzed_types)
        return UnionType.make_union(output, line=t.line)

    def analyze_literal_param(self, idx: int, arg: Type, ctx: Context) -> list[Type] | None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(arg, UnboundType) and arg.original_str_expr is not None:
            assert arg.original_str_fallback is not None
            return [LiteralType(value=arg.original_str_expr, fallback=self.named_type(arg.original_str_fallback), line=arg.line, column=arg.column)]
        if isinstance(arg, UnboundType):
            self.nesting_level += 1
            try:
                arg = self.visit_unbound_type(arg, defining_literal=True)
            finally:
                self.nesting_level -= 1
        arg = get_proper_type(arg)
        if isinstance(arg, AnyType):
            if arg.type_of_any not in (TypeOfAny.from_error, TypeOfAny.special_form):
                self.fail(f'Parameter {idx} of Literal[...] cannot be of type "Any"', ctx, code=codes.VALID_TYPE)
            return None
        elif isinstance(arg, RawExpressionType):
            if arg.literal_value is None:
                name = arg.simple_name()
                if name in ('float', 'complex'):
                    msg = f'Parameter {idx} of Literal[...] cannot be of type "{name}"'
                else:
                    msg = 'Invalid type: Literal[...] cannot contain arbitrary expressions'
                self.fail(msg, ctx, code=codes.VALID_TYPE)
                return None
            fallback = self.named_type(arg.base_type_name)
            assert isinstance(fallback, Instance)
            return [LiteralType(arg.literal_value, fallback, line=arg.line, column=arg.column)]
        elif isinstance(arg, (NoneType, LiteralType)):
            return [arg]
        elif isinstance(arg, Instance) and arg.last_known_value is not None:
            return [arg.last_known_value]
        elif isinstance(arg, UnionType):
            out = []
            for union_arg in arg.items:
                union_result = self.analyze_literal_param(idx, union_arg, ctx)
                if union_result is None:
                    return None
                out.extend(union_result)
            return out
        else:
            self.fail(f'Parameter {idx} of Literal[...] is invalid', ctx, code=codes.VALID_TYPE)
            return None

    def analyze_type(self, t: Type) -> Type:
        if False:
            while True:
                i = 10
        return t.accept(self)

    def fail(self, msg: str, ctx: Context, *, code: ErrorCode | None=None) -> None:
        if False:
            i = 10
            return i + 15
        self.fail_func(msg, ctx, code=code)

    def note(self, msg: str, ctx: Context, *, code: ErrorCode | None=None) -> None:
        if False:
            i = 10
            return i + 15
        self.note_func(msg, ctx, code=code)

    @contextmanager
    def tvar_scope_frame(self) -> Iterator[None]:
        if False:
            for i in range(10):
                print('nop')
        old_scope = self.tvar_scope
        self.tvar_scope = self.tvar_scope.method_frame()
        yield
        self.tvar_scope = old_scope

    def find_type_var_likes(self, t: Type, include_callables: bool=True) -> TypeVarLikeList:
        if False:
            for i in range(10):
                print('nop')
        return t.accept(TypeVarLikeQuery(self.api, self.tvar_scope, include_callables=include_callables))

    def infer_type_variables(self, type: CallableType) -> list[tuple[str, TypeVarLikeExpr]]:
        if False:
            i = 10
            return i + 15
        'Return list of unique type variables referred to in a callable.'
        names: list[str] = []
        tvars: list[TypeVarLikeExpr] = []
        for arg in type.arg_types:
            for (name, tvar_expr) in self.find_type_var_likes(arg):
                if name not in names:
                    names.append(name)
                    tvars.append(tvar_expr)
        for (name, tvar_expr) in self.find_type_var_likes(type.ret_type, include_callables=False):
            if name not in names:
                names.append(name)
                tvars.append(tvar_expr)
        if not names:
            return []
        return list(zip(names, tvars))

    def bind_function_type_variables(self, fun_type: CallableType, defn: Context) -> tuple[Sequence[TypeVarLikeType], bool]:
        if False:
            for i in range(10):
                print('nop')
        'Find the type variables of the function type and bind them in our tvar_scope'
        has_self_type = False
        if fun_type.variables:
            defs = []
            for var in fun_type.variables:
                if self.api.type and self.api.type.self_type and (var == self.api.type.self_type):
                    has_self_type = True
                    continue
                var_node = self.lookup_qualified(var.name, defn)
                assert var_node, 'Binding for function type variable not found within function'
                var_expr = var_node.node
                assert isinstance(var_expr, TypeVarLikeExpr)
                binding = self.tvar_scope.bind_new(var.name, var_expr)
                defs.append(binding)
            return (defs, has_self_type)
        typevars = self.infer_type_variables(fun_type)
        has_self_type = find_self_type(fun_type, lambda name: self.api.lookup_qualified(name, defn, suppress_errors=True))
        typevars = [(name, tvar) for (name, tvar) in typevars if not self.is_defined_type_var(name, defn)]
        defs = []
        for (name, tvar) in typevars:
            if not self.tvar_scope.allow_binding(tvar.fullname):
                self.fail(f'Type variable "{name}" is bound by an outer class', defn, code=codes.VALID_TYPE)
            binding = self.tvar_scope.bind_new(name, tvar)
            defs.append(binding)
        return (defs, has_self_type)

    def is_defined_type_var(self, tvar: str, context: Context) -> bool:
        if False:
            for i in range(10):
                print('nop')
        tvar_node = self.lookup_qualified(tvar, context)
        if not tvar_node:
            return False
        return self.tvar_scope.get_binding(tvar_node) is not None

    def anal_array(self, a: Iterable[Type], nested: bool=True, *, allow_param_spec: bool=False, allow_param_spec_literals: bool=False, allow_unpack: bool=False) -> list[Type]:
        if False:
            for i in range(10):
                print('nop')
        old_allow_param_spec_literals = self.allow_param_spec_literals
        self.allow_param_spec_literals = allow_param_spec_literals
        res: list[Type] = []
        for t in a:
            res.append(self.anal_type(t, nested, allow_param_spec=allow_param_spec, allow_unpack=allow_unpack))
        self.allow_param_spec_literals = old_allow_param_spec_literals
        return self.check_unpacks_in_list(res)

    def anal_type(self, t: Type, nested: bool=True, *, allow_param_spec: bool=False, allow_unpack: bool=False, allow_ellipsis: bool=False) -> Type:
        if False:
            for i in range(10):
                print('nop')
        if nested:
            self.nesting_level += 1
        old_allow_required = self.allow_required
        self.allow_required = False
        old_allow_ellipsis = self.allow_ellipsis
        self.allow_ellipsis = allow_ellipsis
        old_allow_unpack = self.allow_unpack
        self.allow_unpack = allow_unpack
        try:
            analyzed = t.accept(self)
        finally:
            if nested:
                self.nesting_level -= 1
            self.allow_required = old_allow_required
            self.allow_ellipsis = old_allow_ellipsis
            self.allow_unpack = old_allow_unpack
        if not allow_param_spec and isinstance(analyzed, ParamSpecType) and (analyzed.flavor == ParamSpecFlavor.BARE):
            if analyzed.prefix.arg_types:
                self.fail('Invalid location for Concatenate', t, code=codes.VALID_TYPE)
                self.note('You can use Concatenate as the first argument to Callable', t)
                analyzed = AnyType(TypeOfAny.from_error)
            else:
                self.fail(f'Invalid location for ParamSpec "{analyzed.name}"', t, code=codes.VALID_TYPE)
                self.note("You can use ParamSpec as the first argument to Callable, e.g., 'Callable[{}, int]'".format(analyzed.name), t)
                analyzed = AnyType(TypeOfAny.from_error)
        return analyzed

    def anal_var_def(self, var_def: TypeVarLikeType) -> TypeVarLikeType:
        if False:
            print('Hello World!')
        if isinstance(var_def, TypeVarType):
            return TypeVarType(name=var_def.name, fullname=var_def.fullname, id=var_def.id.raw_id, values=self.anal_array(var_def.values), upper_bound=var_def.upper_bound.accept(self), default=var_def.default.accept(self), variance=var_def.variance, line=var_def.line, column=var_def.column)
        else:
            return var_def

    def anal_var_defs(self, var_defs: Sequence[TypeVarLikeType]) -> list[TypeVarLikeType]:
        if False:
            while True:
                i = 10
        return [self.anal_var_def(vd) for vd in var_defs]

    def named_type(self, fully_qualified_name: str, args: list[Type] | None=None, line: int=-1, column: int=-1) -> Instance:
        if False:
            return 10
        node = self.lookup_fully_qualified(fully_qualified_name)
        assert isinstance(node.node, TypeInfo)
        any_type = AnyType(TypeOfAny.special_form)
        if args is not None:
            args = self.check_unpacks_in_list(args)
        return Instance(node.node, args or [any_type] * len(node.node.defn.type_vars), line=line, column=column)

    def check_unpacks_in_list(self, items: list[Type]) -> list[Type]:
        if False:
            for i in range(10):
                print('nop')
        new_items: list[Type] = []
        num_unpacks = 0
        final_unpack = None
        for item in items:
            if isinstance(item, UnpackType) and (not isinstance(get_proper_type(item.type), TupleType)):
                if not num_unpacks:
                    new_items.append(item)
                num_unpacks += 1
                final_unpack = item
            else:
                new_items.append(item)
        if num_unpacks > 1:
            assert final_unpack is not None
            self.fail('More than one Unpack in a type is not allowed', final_unpack)
        return new_items

    def tuple_type(self, items: list[Type], line: int, column: int) -> TupleType:
        if False:
            while True:
                i = 10
        any_type = AnyType(TypeOfAny.special_form)
        return TupleType(items, fallback=self.named_type('builtins.tuple', [any_type]), line=line, column=column)
TypeVarLikeList = List[Tuple[str, TypeVarLikeExpr]]

class MsgCallback(Protocol):

    def __call__(self, __msg: str, __ctx: Context, *, code: ErrorCode | None=None) -> None:
        if False:
            return 10
        ...

def get_omitted_any(disallow_any: bool, fail: MsgCallback, note: MsgCallback, orig_type: Type, options: Options, fullname: str | None=None, unexpanded_type: Type | None=None) -> AnyType:
    if False:
        for i in range(10):
            print('nop')
    if disallow_any:
        nongen_builtins = get_nongen_builtins(options.python_version)
        if fullname in nongen_builtins:
            typ = orig_type
            alternative = nongen_builtins[fullname]
            fail(message_registry.IMPLICIT_GENERIC_ANY_BUILTIN.format(alternative), typ, code=codes.TYPE_ARG)
        else:
            typ = unexpanded_type or orig_type
            type_str = typ.name if isinstance(typ, UnboundType) else format_type_bare(typ, options)
            fail(message_registry.BARE_GENERIC.format(quote_type_string(type_str)), typ, code=codes.TYPE_ARG)
            base_type = get_proper_type(orig_type)
            base_fullname = base_type.type.fullname if isinstance(base_type, Instance) else fullname
            if options.python_version < (3, 9) and base_fullname in GENERIC_STUB_NOT_AT_RUNTIME_TYPES:
                note('Subscripting classes that are not generic at runtime may require escaping, see https://mypy.readthedocs.io/en/stable/runtime_troubles.html#not-generic-runtime', typ, code=codes.TYPE_ARG)
        any_type = AnyType(TypeOfAny.from_error, line=typ.line, column=typ.column)
    else:
        any_type = AnyType(TypeOfAny.from_omitted_generics, line=orig_type.line, column=orig_type.column)
    return any_type

def fix_type_var_tuple_argument(any_type: Type, t: Instance) -> None:
    if False:
        while True:
            i = 10
    if t.type.has_type_var_tuple_type:
        args = list(t.args)
        assert t.type.type_var_tuple_prefix is not None
        tvt = t.type.defn.type_vars[t.type.type_var_tuple_prefix]
        assert isinstance(tvt, TypeVarTupleType)
        args[t.type.type_var_tuple_prefix] = UnpackType(Instance(tvt.tuple_fallback.type, [any_type]))
        t.args = tuple(args)

def fix_instance(t: Instance, fail: MsgCallback, note: MsgCallback, disallow_any: bool, options: Options, use_generic_error: bool=False, unexpanded_type: Type | None=None) -> None:
    if False:
        return 10
    "Fix a malformed instance by replacing all type arguments with Any.\n\n    Also emit a suitable error if this is not due to implicit Any's.\n    "
    if len(t.args) == 0:
        if use_generic_error:
            fullname: str | None = None
        else:
            fullname = t.type.fullname
        any_type = get_omitted_any(disallow_any, fail, note, t, options, fullname, unexpanded_type)
        t.args = (any_type,) * len(t.type.type_vars)
        fix_type_var_tuple_argument(any_type, t)
        return
    any_type = AnyType(TypeOfAny.from_error)
    t.args = tuple((any_type for _ in t.type.type_vars))
    fix_type_var_tuple_argument(any_type, t)
    t.invalid = True

def instantiate_type_alias(node: TypeAlias, args: list[Type], fail: MsgCallback, no_args: bool, ctx: Context, options: Options, *, unexpanded_type: Type | None=None, disallow_any: bool=False, use_standard_error: bool=False, empty_tuple_index: bool=False) -> Type:
    if False:
        print('Hello World!')
    'Create an instance of a (generic) type alias from alias node and type arguments.\n\n    We are following the rules outlined in TypeAlias docstring.\n    Here:\n        node: type alias node (definition)\n        args: type arguments (types to be substituted in place of type variables\n              when expanding the alias)\n        fail: error reporter callback\n        no_args: whether original definition used a bare generic `A = List`\n        ctx: context where expansion happens\n        unexpanded_type, disallow_any, use_standard_error: used to customize error messages\n    '
    args = flatten_nested_tuples(args)
    if any((unknown_unpack(a) for a in args)):
        return set_any_tvars(node, ctx.line, ctx.column, options, special_form=True)
    exp_len = len(node.alias_tvars)
    act_len = len(args)
    if exp_len > 0 and act_len == 0 and (not (empty_tuple_index and node.tvar_tuple_index is not None)):
        return set_any_tvars(node, ctx.line, ctx.column, options, disallow_any=disallow_any, fail=fail, unexpanded_type=unexpanded_type)
    if exp_len == 0 and act_len == 0:
        if no_args:
            assert isinstance(node.target, Instance)
            return Instance(node.target.type, [], line=ctx.line, column=ctx.column)
        return TypeAliasType(node, [], line=ctx.line, column=ctx.column)
    if exp_len == 0 and act_len > 0 and isinstance(node.target, Instance) and no_args:
        tp = Instance(node.target.type, args)
        tp.line = ctx.line
        tp.column = ctx.column
        return tp
    if node.tvar_tuple_index is None:
        if any((isinstance(a, UnpackType) for a in args)):
            fail(message_registry.INVALID_UNPACK_POSITION, ctx, code=codes.VALID_TYPE)
            return set_any_tvars(node, ctx.line, ctx.column, options, from_error=True)
        correct = act_len == exp_len
    else:
        correct = act_len >= exp_len - 1
        for a in args:
            if isinstance(a, UnpackType):
                unpacked = get_proper_type(a.type)
                if isinstance(unpacked, Instance) and unpacked.type.fullname == 'builtins.tuple':
                    correct = True
    if not correct:
        if use_standard_error:
            msg = wrong_type_arg_count(exp_len, str(act_len), node.name)
        else:
            if node.tvar_tuple_index is not None:
                exp_len_str = f'at least {exp_len - 1}'
            else:
                exp_len_str = str(exp_len)
            msg = f'Bad number of arguments for type alias, expected: {exp_len_str}, given: {act_len}'
        fail(msg, ctx, code=codes.TYPE_ARG)
        return set_any_tvars(node, ctx.line, ctx.column, options, from_error=True)
    elif node.tvar_tuple_index is not None:
        unpack = find_unpack_in_list(args)
        if unpack is not None:
            unpack_arg = args[unpack]
            assert isinstance(unpack_arg, UnpackType)
            if isinstance(unpack_arg.type, TypeVarTupleType):
                exp_prefix = node.tvar_tuple_index
                act_prefix = unpack
                exp_suffix = len(node.alias_tvars) - node.tvar_tuple_index - 1
                act_suffix = len(args) - unpack - 1
                if act_prefix < exp_prefix or act_suffix < exp_suffix:
                    fail('TypeVarTuple cannot be split', ctx, code=codes.TYPE_ARG)
                    return set_any_tvars(node, ctx.line, ctx.column, options, from_error=True)
    typ = TypeAliasType(node, args, ctx.line, ctx.column)
    assert typ.alias is not None
    if isinstance(typ.alias.target, Instance) and typ.alias.target.type.fullname == 'mypy_extensions.FlexibleAlias':
        exp = get_proper_type(typ)
        assert isinstance(exp, Instance)
        return exp.args[-1]
    return typ

def set_any_tvars(node: TypeAlias, newline: int, newcolumn: int, options: Options, *, from_error: bool=False, disallow_any: bool=False, special_form: bool=False, fail: MsgCallback | None=None, unexpanded_type: Type | None=None) -> TypeAliasType:
    if False:
        print('Hello World!')
    if from_error or disallow_any:
        type_of_any = TypeOfAny.from_error
    elif special_form:
        type_of_any = TypeOfAny.special_form
    else:
        type_of_any = TypeOfAny.from_omitted_generics
    if disallow_any and node.alias_tvars:
        assert fail is not None
        if unexpanded_type:
            type_str = unexpanded_type.name if isinstance(unexpanded_type, UnboundType) else format_type_bare(unexpanded_type, options)
        else:
            type_str = node.name
        fail(message_registry.BARE_GENERIC.format(quote_type_string(type_str)), Context(newline, newcolumn), code=codes.TYPE_ARG)
    any_type = AnyType(type_of_any, line=newline, column=newcolumn)
    args: list[Type] = []
    for tv in node.alias_tvars:
        if isinstance(tv, TypeVarTupleType):
            args.append(UnpackType(Instance(tv.tuple_fallback.type, [any_type])))
        else:
            args.append(any_type)
    return TypeAliasType(node, args, newline, newcolumn)

def flatten_tvars(lists: list[list[T]]) -> list[T]:
    if False:
        for i in range(10):
            print('nop')
    result: list[T] = []
    for lst in lists:
        for item in lst:
            if item not in result:
                result.append(item)
    return result

class TypeVarLikeQuery(TypeQuery[TypeVarLikeList]):
    """Find TypeVar and ParamSpec references in an unbound type."""

    def __init__(self, api: SemanticAnalyzerCoreInterface, scope: TypeVarLikeScope, *, include_callables: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(flatten_tvars)
        self.api = api
        self.scope = scope
        self.include_callables = include_callables
        self.skip_alias_target = True

    def _seems_like_callable(self, type: UnboundType) -> bool:
        if False:
            i = 10
            return i + 15
        if not type.args:
            return False
        return isinstance(type.args[0], (EllipsisType, TypeList, ParamSpecType))

    def visit_unbound_type(self, t: UnboundType) -> TypeVarLikeList:
        if False:
            for i in range(10):
                print('nop')
        name = t.name
        node = None
        if name.endswith('args'):
            if name.endswith('.args') or name.endswith('.kwargs'):
                base = '.'.join(name.split('.')[:-1])
                n = self.api.lookup_qualified(base, t)
                if n is not None and isinstance(n.node, ParamSpecExpr):
                    node = n
                    name = base
        if node is None:
            node = self.api.lookup_qualified(name, t)
        if node and isinstance(node.node, TypeVarLikeExpr) and (self.scope.get_binding(node) is None):
            assert isinstance(node.node, TypeVarLikeExpr)
            return [(name, node.node)]
        elif not self.include_callables and self._seems_like_callable(t):
            return []
        elif node and node.fullname in LITERAL_TYPE_NAMES:
            return []
        elif node and node.fullname in ANNOTATED_TYPE_NAMES and t.args:
            return self.query_types([t.args[0]])
        else:
            return super().visit_unbound_type(t)

    def visit_callable_type(self, t: CallableType) -> TypeVarLikeList:
        if False:
            return 10
        if self.include_callables:
            return super().visit_callable_type(t)
        else:
            return []

class DivergingAliasDetector(TrivialSyntheticTypeTranslator):
    """See docstring of detect_diverging_alias() for details."""

    def __init__(self, seen_nodes: set[TypeAlias], lookup: Callable[[str, Context], SymbolTableNode | None], scope: TypeVarLikeScope) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.seen_nodes = seen_nodes
        self.lookup = lookup
        self.scope = scope
        self.diverging = False

    def visit_type_alias_type(self, t: TypeAliasType) -> Type:
        if False:
            i = 10
            return i + 15
        assert t.alias is not None, f'Unfixed type alias {t.type_ref}'
        if t.alias in self.seen_nodes:
            for arg in t.args:
                if not (isinstance(arg, TypeVarLikeType) or (isinstance(arg, UnpackType) and isinstance(arg.type, TypeVarLikeType))) and has_type_vars(arg):
                    self.diverging = True
                    return t
            return t
        new_nodes = self.seen_nodes | {t.alias}
        visitor = DivergingAliasDetector(new_nodes, self.lookup, self.scope)
        _ = get_proper_type(t).accept(visitor)
        if visitor.diverging:
            self.diverging = True
        return t

def detect_diverging_alias(node: TypeAlias, target: Type, lookup: Callable[[str, Context], SymbolTableNode | None], scope: TypeVarLikeScope) -> bool:
    if False:
        i = 10
        return i + 15
    "This detects type aliases that will diverge during type checking.\n\n    For example F = Something[..., F[List[T]]]. At each expansion step this will produce\n    *new* type aliases: e.g. F[List[int]], F[List[List[int]]], etc. So we can't detect\n    recursion. It is a known problem in the literature, recursive aliases and generic types\n    don't always go well together. It looks like there is no known systematic solution yet.\n\n    # TODO: should we handle such aliases using type_recursion counter and some large limit?\n    They may be handy in rare cases, e.g. to express a union of non-mixed nested lists:\n    Nested = Union[T, Nested[List[T]]] ~> Union[T, List[T], List[List[T]], ...]\n    "
    visitor = DivergingAliasDetector({node}, lookup, scope)
    _ = target.accept(visitor)
    return visitor.diverging

def check_for_explicit_any(typ: Type | None, options: Options, is_typeshed_stub: bool, msg: MessageBuilder, context: Context) -> None:
    if False:
        print('Hello World!')
    if options.disallow_any_explicit and (not is_typeshed_stub) and typ and has_explicit_any(typ):
        msg.explicit_any(context)

def has_explicit_any(t: Type) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Whether this type is or type it contains is an Any coming from explicit type annotation\n    '
    return t.accept(HasExplicitAny())

class HasExplicitAny(TypeQuery[bool]):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(any)

    def visit_any(self, t: AnyType) -> bool:
        if False:
            while True:
                i = 10
        return t.type_of_any == TypeOfAny.explicit

    def visit_typeddict_type(self, t: TypedDictType) -> bool:
        if False:
            while True:
                i = 10
        return False

def has_any_from_unimported_type(t: Type) -> bool:
    if False:
        i = 10
        return i + 15
    'Return true if this type is Any because an import was not followed.\n\n    If type t is such Any type or has type arguments that contain such Any type\n    this function will return true.\n    '
    return t.accept(HasAnyFromUnimportedType())

class HasAnyFromUnimportedType(BoolTypeQuery):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__(ANY_STRATEGY)

    def visit_any(self, t: AnyType) -> bool:
        if False:
            while True:
                i = 10
        return t.type_of_any == TypeOfAny.from_unimported_type

    def visit_typeddict_type(self, t: TypedDictType) -> bool:
        if False:
            return 10
        return False

def collect_all_inner_types(t: Type) -> list[Type]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return all types that `t` contains\n    '
    return t.accept(CollectAllInnerTypesQuery())

class CollectAllInnerTypesQuery(TypeQuery[List[Type]]):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__(self.combine_lists_strategy)

    def query_types(self, types: Iterable[Type]) -> list[Type]:
        if False:
            for i in range(10):
                print('nop')
        return self.strategy([t.accept(self) for t in types]) + list(types)

    @classmethod
    def combine_lists_strategy(cls, it: Iterable[list[Type]]) -> list[Type]:
        if False:
            i = 10
            return i + 15
        return list(itertools.chain.from_iterable(it))

def make_optional_type(t: Type) -> Type:
    if False:
        for i in range(10):
            print('nop')
    "Return the type corresponding to Optional[t].\n\n    Note that we can't use normal union simplification, since this function\n    is called during semantic analysis and simplification only works during\n    type checking.\n    "
    p_t = get_proper_type(t)
    if isinstance(p_t, NoneType):
        return t
    elif isinstance(p_t, UnionType):
        items = [item for item in flatten_nested_unions(p_t.items, handle_type_alias_type=False) if not isinstance(get_proper_type(item), NoneType)]
        return UnionType(items + [NoneType()], t.line, t.column)
    else:
        return UnionType([t, NoneType()], t.line, t.column)

def validate_instance(t: Instance, fail: MsgCallback, empty_tuple_index: bool) -> bool:
    if False:
        while True:
            i = 10
    'Check if this is a well-formed instance with respect to argument count/positions.'
    if any((unknown_unpack(a) for a in t.args)):
        return False
    if t.type.has_type_var_tuple_type:
        correct = len(t.args) >= len(t.type.type_vars) - 1
        if any((isinstance(a, UnpackType) and isinstance(get_proper_type(a.type), Instance) for a in t.args)):
            correct = True
        if not correct:
            exp_len = f'at least {len(t.type.type_vars) - 1}'
            fail(f'Bad number of arguments, expected: {exp_len}, given: {len(t.args)}', t, code=codes.TYPE_ARG)
            return False
        elif not t.args:
            if not (empty_tuple_index and len(t.type.type_vars) == 1):
                return False
        else:
            unpack = find_unpack_in_list(t.args)
            if unpack is not None:
                unpack_arg = t.args[unpack]
                assert isinstance(unpack_arg, UnpackType)
                if isinstance(unpack_arg.type, TypeVarTupleType):
                    assert t.type.type_var_tuple_prefix is not None
                    assert t.type.type_var_tuple_suffix is not None
                    exp_prefix = t.type.type_var_tuple_prefix
                    act_prefix = unpack
                    exp_suffix = t.type.type_var_tuple_suffix
                    act_suffix = len(t.args) - unpack - 1
                    if act_prefix < exp_prefix or act_suffix < exp_suffix:
                        fail('TypeVarTuple cannot be split', t, code=codes.TYPE_ARG)
                        return False
    elif any((isinstance(a, UnpackType) for a in t.args)):
        fail(message_registry.INVALID_UNPACK_POSITION, t, code=codes.VALID_TYPE)
        return False
    elif len(t.args) != len(t.type.type_vars):
        if t.args:
            fail(wrong_type_arg_count(len(t.type.type_vars), str(len(t.args)), t.type.name), t, code=codes.TYPE_ARG)
        return False
    return True

def find_self_type(typ: Type, lookup: Callable[[str], SymbolTableNode | None]) -> bool:
    if False:
        print('Hello World!')
    return typ.accept(HasSelfType(lookup))

class HasSelfType(BoolTypeQuery):

    def __init__(self, lookup: Callable[[str], SymbolTableNode | None]) -> None:
        if False:
            return 10
        self.lookup = lookup
        super().__init__(ANY_STRATEGY)

    def visit_unbound_type(self, t: UnboundType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        sym = self.lookup(t.name)
        if sym and sym.fullname in SELF_TYPE_NAMES:
            return True
        return super().visit_unbound_type(t)

def unknown_unpack(t: Type) -> bool:
    if False:
        i = 10
        return i + 15
    'Check if a given type is an unpack of an unknown type.\n\n    Unfortunately, there is no robust way to distinguish forward references from\n    genuine undefined names here. But this worked well so far, although it looks\n    quite fragile.\n    '
    if isinstance(t, UnpackType):
        unpacked = get_proper_type(t.type)
        if isinstance(unpacked, AnyType) and unpacked.type_of_any == TypeOfAny.special_form:
            return True
    return False