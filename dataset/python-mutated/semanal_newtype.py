"""Semantic analysis of NewType definitions.

This is conceptually part of mypy.semanal (semantic analyzer pass 2).
"""
from __future__ import annotations
from mypy import errorcodes as codes
from mypy.errorcodes import ErrorCode
from mypy.exprtotype import TypeTranslationError, expr_to_unanalyzed_type
from mypy.messages import MessageBuilder, format_type
from mypy.nodes import ARG_POS, MDEF, Argument, AssignmentStmt, Block, CallExpr, Context, FuncDef, NameExpr, NewTypeExpr, PlaceholderNode, RefExpr, StrExpr, SymbolTableNode, TypeInfo, Var
from mypy.options import Options
from mypy.semanal_shared import SemanticAnalyzerInterface, has_placeholder
from mypy.typeanal import check_for_explicit_any, has_any_from_unimported_type
from mypy.types import AnyType, CallableType, Instance, NoneType, PlaceholderType, TupleType, Type, TypeOfAny, get_proper_type

class NewTypeAnalyzer:

    def __init__(self, options: Options, api: SemanticAnalyzerInterface, msg: MessageBuilder) -> None:
        if False:
            i = 10
            return i + 15
        self.options = options
        self.api = api
        self.msg = msg

    def process_newtype_declaration(self, s: AssignmentStmt) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Check if s declares a NewType; if yes, store it in symbol table.\n\n        Return True if it's a NewType declaration. The current target may be\n        deferred as a side effect if the base type is not ready, even if\n        the return value is True.\n\n        The logic in this function mostly copies the logic for visit_class_def()\n        with a single (non-Generic) base.\n        "
        (var_name, call) = self.analyze_newtype_declaration(s)
        if var_name is None or call is None:
            return False
        name = var_name
        if self.api.is_func_scope():
            name += '@' + str(s.line)
        fullname = self.api.qualified_name(name)
        if not call.analyzed or (isinstance(call.analyzed, NewTypeExpr) and (not call.analyzed.info)):
            placeholder = PlaceholderNode(fullname, s, s.line, becomes_typeinfo=True)
            self.api.add_symbol(var_name, placeholder, s, can_defer=False)
        (old_type, should_defer) = self.check_newtype_args(var_name, call, s)
        old_type = get_proper_type(old_type)
        if not isinstance(call.analyzed, NewTypeExpr):
            call.analyzed = NewTypeExpr(var_name, old_type, line=call.line, column=call.column)
        else:
            call.analyzed.old_type = old_type
        if old_type is None:
            if should_defer:
                self.api.defer()
                return True
        assert isinstance(call.analyzed, NewTypeExpr)
        if isinstance(old_type, TupleType):
            newtype_class_info = self.build_newtype_typeinfo(name, old_type, old_type.partial_fallback, s.line, call.analyzed.info)
            newtype_class_info.update_tuple_type(old_type)
        elif isinstance(old_type, Instance):
            if old_type.type.is_protocol:
                self.fail('NewType cannot be used with protocol classes', s)
            newtype_class_info = self.build_newtype_typeinfo(name, old_type, old_type, s.line, call.analyzed.info)
        else:
            if old_type is not None:
                message = 'Argument 2 to NewType(...) must be subclassable (got {})'
                self.fail(message.format(format_type(old_type, self.options)), s, code=codes.VALID_NEWTYPE)
            old_type = AnyType(TypeOfAny.from_error)
            object_type = self.api.named_type('builtins.object')
            newtype_class_info = self.build_newtype_typeinfo(name, old_type, object_type, s.line, call.analyzed.info)
            newtype_class_info.fallback_to_any = True
        check_for_explicit_any(old_type, self.options, self.api.is_typeshed_stub_file, self.msg, context=s)
        if self.options.disallow_any_unimported and has_any_from_unimported_type(old_type):
            self.msg.unimported_type_becomes_any('Argument 2 to NewType(...)', old_type, s)
        assert isinstance(call.analyzed, NewTypeExpr)
        if not call.analyzed.info:
            call.analyzed.info = newtype_class_info
        else:
            call.analyzed.info.bases = newtype_class_info.bases
        self.api.add_symbol(var_name, call.analyzed.info, s)
        if self.api.is_func_scope():
            self.api.add_symbol_skip_local(name, call.analyzed.info)
        newtype_class_info.line = s.line
        return True

    def analyze_newtype_declaration(self, s: AssignmentStmt) -> tuple[str | None, CallExpr | None]:
        if False:
            for i in range(10):
                print('nop')
        'Return the NewType call expression if `s` is a newtype declaration or None otherwise.'
        (name, call) = (None, None)
        if len(s.lvalues) == 1 and isinstance(s.lvalues[0], NameExpr) and isinstance(s.rvalue, CallExpr) and isinstance(s.rvalue.callee, RefExpr) and (s.rvalue.callee.fullname in ('typing.NewType', 'typing_extensions.NewType')):
            name = s.lvalues[0].name
            if s.type:
                self.fail('Cannot declare the type of a NewType declaration', s)
            names = self.api.current_symbol_table()
            existing = names.get(name)
            if existing and (not isinstance(existing.node, PlaceholderNode)) and (not s.rvalue.analyzed):
                self.fail(f'Cannot redefine "{name}" as a NewType', s)
            call = s.rvalue
        return (name, call)

    def check_newtype_args(self, name: str, call: CallExpr, context: Context) -> tuple[Type | None, bool]:
        if False:
            while True:
                i = 10
        'Ananlyze base type in NewType call.\n\n        Return a tuple (type, should defer).\n        '
        has_failed = False
        (args, arg_kinds) = (call.args, call.arg_kinds)
        if len(args) != 2 or arg_kinds[0] != ARG_POS or arg_kinds[1] != ARG_POS:
            self.fail('NewType(...) expects exactly two positional arguments', context)
            return (None, False)
        if not isinstance(args[0], StrExpr):
            self.fail('Argument 1 to NewType(...) must be a string literal', context)
            has_failed = True
        elif args[0].value != name:
            msg = 'String argument 1 "{}" to NewType(...) does not match variable name "{}"'
            self.fail(msg.format(args[0].value, name), context)
            has_failed = True
        msg = 'Argument 2 to NewType(...) must be a valid type'
        try:
            unanalyzed_type = expr_to_unanalyzed_type(args[1], self.options, self.api.is_stub_file)
        except TypeTranslationError:
            self.fail(msg, context)
            return (None, False)
        old_type = get_proper_type(self.api.anal_type(unanalyzed_type, report_invalid_types=False, allow_placeholder=not self.api.is_func_scope()))
        should_defer = False
        if isinstance(old_type, PlaceholderType):
            old_type = None
        if old_type is None:
            should_defer = True
        if isinstance(old_type, AnyType) and old_type.is_from_error:
            self.fail(msg, context)
            return (None, False)
        return (None if has_failed else old_type, should_defer)

    def build_newtype_typeinfo(self, name: str, old_type: Type, base_type: Instance, line: int, existing_info: TypeInfo | None) -> TypeInfo:
        if False:
            print('Hello World!')
        info = existing_info or self.api.basic_new_typeinfo(name, base_type, line)
        info.bases = [base_type]
        info.is_newtype = True
        args = [Argument(Var('self'), NoneType(), None, ARG_POS), self.make_argument('item', old_type)]
        signature = CallableType(arg_types=[Instance(info, []), old_type], arg_kinds=[arg.kind for arg in args], arg_names=['self', 'item'], ret_type=NoneType(), fallback=self.api.named_type('builtins.function'), name=name)
        init_func = FuncDef('__init__', args, Block([]), typ=signature)
        init_func.info = info
        init_func._fullname = info.fullname + '.__init__'
        if not existing_info:
            updated = True
        else:
            previous_sym = info.names['__init__'].node
            assert isinstance(previous_sym, FuncDef)
            updated = old_type != previous_sym.arguments[1].variable.type
        info.names['__init__'] = SymbolTableNode(MDEF, init_func)
        if has_placeholder(old_type):
            self.api.process_placeholder(None, 'NewType base', info, force_progress=updated)
        return info

    def make_argument(self, name: str, type: Type) -> Argument:
        if False:
            i = 10
            return i + 15
        return Argument(Var(name), type, None, ARG_POS)

    def fail(self, msg: str, ctx: Context, *, code: ErrorCode | None=None) -> None:
        if False:
            print('Hello World!')
        self.api.fail(msg, ctx, code=code)