"""Semantic analysis of named tuple definitions.

This is conceptually part of mypy.semanal.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Final, Iterator, List, Mapping, cast
from mypy.exprtotype import TypeTranslationError, expr_to_unanalyzed_type
from mypy.nodes import ARG_NAMED_OPT, ARG_OPT, ARG_POS, MDEF, Argument, AssignmentStmt, Block, CallExpr, ClassDef, Context, Decorator, EllipsisExpr, Expression, ExpressionStmt, FuncBase, FuncDef, ListExpr, NamedTupleExpr, NameExpr, PassStmt, RefExpr, Statement, StrExpr, SymbolTable, SymbolTableNode, TempNode, TupleExpr, TypeInfo, TypeVarExpr, Var, is_StrExpr_list
from mypy.options import Options
from mypy.semanal_shared import PRIORITY_FALLBACKS, SemanticAnalyzerInterface, calculate_tuple_fallback, has_placeholder, set_callable_name
from mypy.types import TYPED_NAMEDTUPLE_NAMES, AnyType, CallableType, LiteralType, TupleType, Type, TypeOfAny, TypeType, TypeVarLikeType, TypeVarType, UnboundType, has_type_vars
from mypy.util import get_unique_redefinition_name
NAMEDTUPLE_PROHIBITED_NAMES: Final = ('__new__', '__init__', '__slots__', '__getnewargs__', '_fields', '_field_defaults', '_field_types', '_make', '_replace', '_asdict', '_source', '__annotations__')
NAMEDTUP_CLASS_ERROR: Final = 'Invalid statement in NamedTuple definition; expected "field_name: field_type [= default]"'
SELF_TVAR_NAME: Final = '_NT'

class NamedTupleAnalyzer:

    def __init__(self, options: Options, api: SemanticAnalyzerInterface) -> None:
        if False:
            print('Hello World!')
        self.options = options
        self.api = api

    def analyze_namedtuple_classdef(self, defn: ClassDef, is_stub_file: bool, is_func_scope: bool) -> tuple[bool, TypeInfo | None]:
        if False:
            while True:
                i = 10
        'Analyze if given class definition can be a named tuple definition.\n\n        Return a tuple where first item indicates whether this can possibly be a named tuple,\n        and the second item is the corresponding TypeInfo (may be None if not ready and should be\n        deferred).\n        '
        for base_expr in defn.base_type_exprs:
            if isinstance(base_expr, RefExpr):
                self.api.accept(base_expr)
                if base_expr.fullname in TYPED_NAMEDTUPLE_NAMES:
                    result = self.check_namedtuple_classdef(defn, is_stub_file)
                    if result is None:
                        return (True, None)
                    (items, types, default_items, statements) = result
                    if is_func_scope and '@' not in defn.name:
                        defn.name += '@' + str(defn.line)
                    existing_info = None
                    if isinstance(defn.analyzed, NamedTupleExpr):
                        existing_info = defn.analyzed.info
                    info = self.build_namedtuple_typeinfo(defn.name, items, types, default_items, defn.line, existing_info)
                    defn.analyzed = NamedTupleExpr(info, is_typed=True)
                    defn.analyzed.line = defn.line
                    defn.analyzed.column = defn.column
                    defn.defs.body = statements
                    return (True, info)
        return (False, None)

    def check_namedtuple_classdef(self, defn: ClassDef, is_stub_file: bool) -> tuple[list[str], list[Type], dict[str, Expression], list[Statement]] | None:
        if False:
            return 10
        'Parse and validate fields in named tuple class definition.\n\n        Return a four tuple:\n          * field names\n          * field types\n          * field default values\n          * valid statements\n        or None, if any of the types are not ready.\n        '
        if len(defn.base_type_exprs) > 1:
            self.fail('NamedTuple should be a single base', defn)
        items: list[str] = []
        types: list[Type] = []
        default_items: dict[str, Expression] = {}
        statements: list[Statement] = []
        for stmt in defn.defs.body:
            statements.append(stmt)
            if not isinstance(stmt, AssignmentStmt):
                if isinstance(stmt, PassStmt) or (isinstance(stmt, ExpressionStmt) and isinstance(stmt.expr, EllipsisExpr)):
                    continue
                if isinstance(stmt, (Decorator, FuncBase)):
                    continue
                if isinstance(stmt, ExpressionStmt) and isinstance(stmt.expr, StrExpr):
                    continue
                statements.pop()
                defn.removed_statements.append(stmt)
                self.fail(NAMEDTUP_CLASS_ERROR, stmt)
            elif len(stmt.lvalues) > 1 or not isinstance(stmt.lvalues[0], NameExpr):
                statements.pop()
                defn.removed_statements.append(stmt)
                self.fail(NAMEDTUP_CLASS_ERROR, stmt)
            else:
                name = stmt.lvalues[0].name
                items.append(name)
                if stmt.type is None:
                    types.append(AnyType(TypeOfAny.unannotated))
                else:
                    analyzed = self.api.anal_type(stmt.type, allow_placeholder=not self.api.is_func_scope(), prohibit_self_type='NamedTuple item type')
                    if analyzed is None:
                        return None
                    types.append(analyzed)
                if name.startswith('_'):
                    self.fail(f'NamedTuple field name cannot start with an underscore: {name}', stmt)
                if stmt.type is None or (hasattr(stmt, 'new_syntax') and (not stmt.new_syntax)):
                    self.fail(NAMEDTUP_CLASS_ERROR, stmt)
                elif isinstance(stmt.rvalue, TempNode):
                    if default_items:
                        self.fail('Non-default NamedTuple fields cannot follow default fields', stmt)
                else:
                    default_items[name] = stmt.rvalue
        return (items, types, default_items, statements)

    def check_namedtuple(self, node: Expression, var_name: str | None, is_func_scope: bool) -> tuple[str | None, TypeInfo | None, list[TypeVarLikeType]]:
        if False:
            print('Hello World!')
        'Check if a call defines a namedtuple.\n\n        The optional var_name argument is the name of the variable to\n        which this is assigned, if any.\n\n        Return a tuple of two items:\n          * Internal name of the named tuple (e.g. the name passed as an argument to namedtuple)\n            or None if it is not a valid named tuple\n          * Corresponding TypeInfo, or None if not ready.\n\n        If the definition is invalid but looks like a namedtuple,\n        report errors but return (some) TypeInfo.\n        '
        if not isinstance(node, CallExpr):
            return (None, None, [])
        call = node
        callee = call.callee
        if not isinstance(callee, RefExpr):
            return (None, None, [])
        fullname = callee.fullname
        if fullname == 'collections.namedtuple':
            is_typed = False
        elif fullname in TYPED_NAMEDTUPLE_NAMES:
            is_typed = True
        else:
            return (None, None, [])
        result = self.parse_namedtuple_args(call, fullname)
        if result:
            (items, types, defaults, typename, tvar_defs, ok) = result
        else:
            if var_name:
                name = var_name
                if is_func_scope:
                    name += '@' + str(call.line)
            else:
                name = var_name = 'namedtuple@' + str(call.line)
            info = self.build_namedtuple_typeinfo(name, [], [], {}, node.line, None)
            self.store_namedtuple_info(info, var_name, call, is_typed)
            if name != var_name or is_func_scope:
                self.api.add_symbol_skip_local(name, info)
            return (var_name, info, [])
        if not ok:
            return (typename, None, [])
        if var_name:
            name = var_name
        else:
            name = typename
        if var_name is None or is_func_scope:
            name += '@' + str(call.line)
        if defaults:
            default_items = {arg_name: default for (arg_name, default) in zip(items[-len(defaults):], defaults)}
        else:
            default_items = {}
        existing_info = None
        if isinstance(node.analyzed, NamedTupleExpr):
            existing_info = node.analyzed.info
        info = self.build_namedtuple_typeinfo(name, items, types, default_items, node.line, existing_info)
        if var_name:
            self.store_namedtuple_info(info, var_name, call, is_typed)
        else:
            call.analyzed = NamedTupleExpr(info, is_typed=is_typed)
            call.analyzed.set_line(call)
        if name != var_name or is_func_scope:
            self.api.add_symbol_skip_local(name, info)
        return (typename, info, tvar_defs)

    def store_namedtuple_info(self, info: TypeInfo, name: str, call: CallExpr, is_typed: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.api.add_symbol(name, info, call)
        call.analyzed = NamedTupleExpr(info, is_typed=is_typed)
        call.analyzed.set_line(call)

    def parse_namedtuple_args(self, call: CallExpr, fullname: str) -> None | tuple[list[str], list[Type], list[Expression], str, list[TypeVarLikeType], bool]:
        if False:
            for i in range(10):
                print('nop')
        "Parse a namedtuple() call into data needed to construct a type.\n\n        Returns a 6-tuple:\n        - List of argument names\n        - List of argument types\n        - List of default values\n        - First argument of namedtuple\n        - All typevars found in the field definition\n        - Whether all types are ready.\n\n        Return None if the definition didn't typecheck.\n        "
        type_name = 'NamedTuple' if fullname in TYPED_NAMEDTUPLE_NAMES else 'namedtuple'
        args = call.args
        if len(args) < 2:
            self.fail(f'Too few arguments for "{type_name}()"', call)
            return None
        defaults: list[Expression] = []
        if len(args) > 2:
            if fullname in TYPED_NAMEDTUPLE_NAMES:
                self.fail('Too many arguments for "NamedTuple()"', call)
                return None
            for (i, arg_name) in enumerate(call.arg_names[2:], 2):
                if arg_name == 'defaults':
                    arg = args[i]
                    if isinstance(arg, (ListExpr, TupleExpr)):
                        defaults = list(arg.items)
                    else:
                        self.fail('List or tuple literal expected as the defaults argument to {}()'.format(type_name), arg)
                    break
        if call.arg_kinds[:2] != [ARG_POS, ARG_POS]:
            self.fail(f'Unexpected arguments to "{type_name}()"', call)
            return None
        if not isinstance(args[0], StrExpr):
            self.fail(f'"{type_name}()" expects a string literal as the first argument', call)
            return None
        typename = args[0].value
        types: list[Type] = []
        tvar_defs = []
        if not isinstance(args[1], (ListExpr, TupleExpr)):
            if fullname == 'collections.namedtuple' and isinstance(args[1], StrExpr):
                str_expr = args[1]
                items = str_expr.value.replace(',', ' ').split()
            else:
                self.fail('List or tuple literal expected as the second argument to "{}()"'.format(type_name), call)
                return None
        else:
            listexpr = args[1]
            if fullname == 'collections.namedtuple':
                if not is_StrExpr_list(listexpr.items):
                    self.fail('String literal expected as "namedtuple()" item', call)
                    return None
                items = [item.value for item in listexpr.items]
            else:
                type_exprs = [t.items[1] for t in listexpr.items if isinstance(t, TupleExpr) and len(t.items) == 2]
                tvar_defs = self.api.get_and_bind_all_tvars(type_exprs)
                result = self.parse_namedtuple_fields_with_types(listexpr.items, call)
                if result is None:
                    return None
                (items, types, _, ok) = result
                if not ok:
                    return ([], [], [], typename, [], False)
        if not types:
            types = [AnyType(TypeOfAny.unannotated) for _ in items]
        underscore = [item for item in items if item.startswith('_')]
        if underscore:
            self.fail(f'"{type_name}()" field names cannot start with an underscore: ' + ', '.join(underscore), call)
        if len(defaults) > len(items):
            self.fail(f'Too many defaults given in call to "{type_name}()"', call)
            defaults = defaults[:len(items)]
        return (items, types, defaults, typename, tvar_defs, True)

    def parse_namedtuple_fields_with_types(self, nodes: list[Expression], context: Context) -> tuple[list[str], list[Type], list[Expression], bool] | None:
        if False:
            while True:
                i = 10
        'Parse typed named tuple fields.\n\n        Return (names, types, defaults, whether types are all ready), or None if error occurred.\n        '
        items: list[str] = []
        types: list[Type] = []
        for item in nodes:
            if isinstance(item, TupleExpr):
                if len(item.items) != 2:
                    self.fail('Invalid "NamedTuple()" field definition', item)
                    return None
                (name, type_node) = item.items
                if isinstance(name, StrExpr):
                    items.append(name.value)
                else:
                    self.fail('Invalid "NamedTuple()" field name', item)
                    return None
                try:
                    type = expr_to_unanalyzed_type(type_node, self.options, self.api.is_stub_file)
                except TypeTranslationError:
                    self.fail('Invalid field type', type_node)
                    return None
                analyzed = self.api.anal_type(type, allow_placeholder=not self.api.is_func_scope(), prohibit_self_type='NamedTuple item type')
                if isinstance(analyzed, UnboundType):
                    analyzed = AnyType(TypeOfAny.from_error)
                if analyzed is None:
                    return ([], [], [], False)
                types.append(analyzed)
            else:
                self.fail('Tuple expected as "NamedTuple()" field', item)
                return None
        return (items, types, [], True)

    def build_namedtuple_typeinfo(self, name: str, items: list[str], types: list[Type], default_items: Mapping[str, Expression], line: int, existing_info: TypeInfo | None) -> TypeInfo:
        if False:
            print('Hello World!')
        strtype = self.api.named_type('builtins.str')
        implicit_any = AnyType(TypeOfAny.special_form)
        basetuple_type = self.api.named_type('builtins.tuple', [implicit_any])
        dictype = self.api.named_type('builtins.dict', [strtype, implicit_any])
        ordereddictype = self.api.named_type('builtins.dict', [strtype, implicit_any])
        fallback = self.api.named_type('builtins.tuple', [implicit_any])
        iterable_type = self.api.named_type_or_none('typing.Iterable', [implicit_any])
        function_type = self.api.named_type('builtins.function')
        literals: list[Type] = [LiteralType(item, strtype) for item in items]
        match_args_type = TupleType(literals, basetuple_type)
        info = existing_info or self.api.basic_new_typeinfo(name, fallback, line)
        info.is_named_tuple = True
        tuple_base = TupleType(types, fallback)
        if info.special_alias and has_placeholder(info.special_alias.target):
            self.api.process_placeholder(None, 'NamedTuple item', info, force_progress=tuple_base != info.tuple_type)
        info.update_tuple_type(tuple_base)
        info.line = line
        info.metadata['namedtuple'] = {'fields': items.copy()}
        if not has_placeholder(tuple_base) and (not has_type_vars(tuple_base)):
            self.api.schedule_patch(PRIORITY_FALLBACKS, lambda : calculate_tuple_fallback(tuple_base))

        def add_field(var: Var, is_initialized_in_class: bool=False, is_property: bool=False) -> None:
            if False:
                for i in range(10):
                    print('nop')
            var.info = info
            var.is_initialized_in_class = is_initialized_in_class
            var.is_property = is_property
            var._fullname = f'{info.fullname}.{var.name}'
            info.names[var.name] = SymbolTableNode(MDEF, var)
        fields = [Var(item, typ) for (item, typ) in zip(items, types)]
        for var in fields:
            add_field(var, is_property=True)
        vars = [Var(item, typ) for (item, typ) in zip(items, types)]
        tuple_of_strings = TupleType([strtype for _ in items], basetuple_type)
        add_field(Var('_fields', tuple_of_strings), is_initialized_in_class=True)
        add_field(Var('_field_types', dictype), is_initialized_in_class=True)
        add_field(Var('_field_defaults', dictype), is_initialized_in_class=True)
        add_field(Var('_source', strtype), is_initialized_in_class=True)
        add_field(Var('__annotations__', ordereddictype), is_initialized_in_class=True)
        add_field(Var('__doc__', strtype), is_initialized_in_class=True)
        if self.options.python_version >= (3, 10):
            add_field(Var('__match_args__', match_args_type), is_initialized_in_class=True)
        assert info.tuple_type is not None
        tvd = TypeVarType(name=SELF_TVAR_NAME, fullname=info.fullname + '.' + SELF_TVAR_NAME, id=self.api.tvar_scope.new_unique_func_id(), values=[], upper_bound=info.tuple_type, default=AnyType(TypeOfAny.from_omitted_generics))
        selftype = tvd

        def add_method(funcname: str, ret: Type, args: list[Argument], is_classmethod: bool=False, is_new: bool=False) -> None:
            if False:
                print('Hello World!')
            if is_classmethod or is_new:
                first = [Argument(Var('_cls'), TypeType.make_normalized(selftype), None, ARG_POS)]
            else:
                first = [Argument(Var('_self'), selftype, None, ARG_POS)]
            args = first + args
            types = [arg.type_annotation for arg in args]
            items = [arg.variable.name for arg in args]
            arg_kinds = [arg.kind for arg in args]
            assert None not in types
            signature = CallableType(cast(List[Type], types), arg_kinds, items, ret, function_type)
            signature.variables = [tvd]
            func = FuncDef(funcname, args, Block([]))
            func.info = info
            func.is_class = is_classmethod
            func.type = set_callable_name(signature, func)
            func._fullname = info.fullname + '.' + funcname
            func.line = line
            if is_classmethod:
                v = Var(funcname, func.type)
                v.is_classmethod = True
                v.info = info
                v._fullname = func._fullname
                func.is_decorated = True
                dec = Decorator(func, [NameExpr('classmethod')], v)
                dec.line = line
                sym = SymbolTableNode(MDEF, dec)
            else:
                sym = SymbolTableNode(MDEF, func)
            sym.plugin_generated = True
            info.names[funcname] = sym
        add_method('_replace', ret=selftype, args=[Argument(var, var.type, EllipsisExpr(), ARG_NAMED_OPT) for var in vars])

        def make_init_arg(var: Var) -> Argument:
            if False:
                for i in range(10):
                    print('nop')
            default = default_items.get(var.name, None)
            kind = ARG_POS if default is None else ARG_OPT
            return Argument(var, var.type, default, kind)
        add_method('__new__', ret=selftype, args=[make_init_arg(var) for var in vars], is_new=True)
        add_method('_asdict', args=[], ret=ordereddictype)
        add_method('_make', ret=selftype, is_classmethod=True, args=[Argument(Var('iterable', iterable_type), iterable_type, None, ARG_POS)])
        self_tvar_expr = TypeVarExpr(SELF_TVAR_NAME, info.fullname + '.' + SELF_TVAR_NAME, [], info.tuple_type, AnyType(TypeOfAny.from_omitted_generics))
        info.names[SELF_TVAR_NAME] = SymbolTableNode(MDEF, self_tvar_expr)
        return info

    @contextmanager
    def save_namedtuple_body(self, named_tuple_info: TypeInfo) -> Iterator[None]:
        if False:
            return 10
        "Preserve the generated body of class-based named tuple and then restore it.\n\n        Temporarily clear the names dict so we don't get errors about duplicate names\n        that were already set in build_namedtuple_typeinfo (we already added the tuple\n        field names while generating the TypeInfo, and actual duplicates are\n        already reported).\n        "
        nt_names = named_tuple_info.names
        named_tuple_info.names = SymbolTable()
        yield
        for prohibited in NAMEDTUPLE_PROHIBITED_NAMES:
            if prohibited in named_tuple_info.names:
                if nt_names.get(prohibited) is named_tuple_info.names[prohibited]:
                    continue
                ctx = named_tuple_info.names[prohibited].node
                assert ctx is not None
                self.fail(f'Cannot overwrite NamedTuple attribute "{prohibited}"', ctx)
        for (key, value) in nt_names.items():
            if key in named_tuple_info.names:
                if key == '__doc__':
                    continue
                sym = named_tuple_info.names[key]
                if isinstance(sym.node, (FuncBase, Decorator)) and (not sym.plugin_generated):
                    continue
                r_key = get_unique_redefinition_name(key, named_tuple_info.names)
                named_tuple_info.names[r_key] = sym
            named_tuple_info.names[key] = value

    def fail(self, msg: str, ctx: Context) -> None:
        if False:
            while True:
                i = 10
        self.api.fail(msg, ctx)