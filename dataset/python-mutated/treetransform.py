"""Base visitor that implements an identity AST transform.

Subclass TransformVisitor to perform non-trivial transformations.
"""
from __future__ import annotations
from typing import Iterable, Optional, cast
from mypy.nodes import GDEF, REVEAL_TYPE, Argument, AssertStmt, AssertTypeExpr, AssignmentExpr, AssignmentStmt, AwaitExpr, Block, BreakStmt, BytesExpr, CallExpr, CastExpr, ClassDef, ComparisonExpr, ComplexExpr, ConditionalExpr, ContinueStmt, Decorator, DelStmt, DictExpr, DictionaryComprehension, EllipsisExpr, EnumCallExpr, Expression, ExpressionStmt, FloatExpr, ForStmt, FuncDef, FuncItem, GeneratorExpr, GlobalDecl, IfStmt, Import, ImportAll, ImportFrom, IndexExpr, IntExpr, LambdaExpr, ListComprehension, ListExpr, MatchStmt, MemberExpr, MypyFile, NamedTupleExpr, NameExpr, NewTypeExpr, Node, NonlocalDecl, OperatorAssignmentStmt, OpExpr, OverloadedFuncDef, OverloadPart, ParamSpecExpr, PassStmt, PromoteExpr, RaiseStmt, RefExpr, ReturnStmt, RevealExpr, SetComprehension, SetExpr, SliceExpr, StarExpr, Statement, StrExpr, SuperExpr, SymbolTable, TempNode, TryStmt, TupleExpr, TypeAliasExpr, TypeApplication, TypedDictExpr, TypeVarExpr, TypeVarTupleExpr, UnaryExpr, Var, WhileStmt, WithStmt, YieldExpr, YieldFromExpr
from mypy.patterns import AsPattern, ClassPattern, MappingPattern, OrPattern, Pattern, SequencePattern, SingletonPattern, StarredPattern, ValuePattern
from mypy.traverser import TraverserVisitor
from mypy.types import FunctionLike, ProperType, Type
from mypy.util import replace_object_state
from mypy.visitor import NodeVisitor

class TransformVisitor(NodeVisitor[Node]):
    """Transform a semantically analyzed AST (or subtree) to an identical copy.

    Use the node() method to transform an AST node.

    Subclass to perform a non-identity transform.

    Notes:

     * This can only be used to transform functions or classes, not top-level
       statements, and/or modules as a whole.
     * Do not duplicate TypeInfo nodes. This would generally not be desirable.
     * Only update some name binding cross-references, but only those that
       refer to Var, Decorator or FuncDef nodes, not those targeting ClassDef or
       TypeInfo nodes.
     * Types are not transformed, but you can override type() to also perform
       type transformation.

    TODO nested classes and functions have not been tested well enough
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.test_only = False
        self.var_map: dict[Var, Var] = {}
        self.func_placeholder_map: dict[FuncDef, FuncDef] = {}

    def visit_mypy_file(self, node: MypyFile) -> MypyFile:
        if False:
            while True:
                i = 10
        assert self.test_only, 'This visitor should not be used for whole files.'
        ignored_lines = {line: codes.copy() for (line, codes) in node.ignored_lines.items()}
        new = MypyFile(self.statements(node.defs), [], node.is_bom, ignored_lines=ignored_lines)
        new._fullname = node._fullname
        new.path = node.path
        new.names = SymbolTable()
        return new

    def visit_import(self, node: Import) -> Import:
        if False:
            return 10
        return Import(node.ids.copy())

    def visit_import_from(self, node: ImportFrom) -> ImportFrom:
        if False:
            print('Hello World!')
        return ImportFrom(node.id, node.relative, node.names.copy())

    def visit_import_all(self, node: ImportAll) -> ImportAll:
        if False:
            for i in range(10):
                print('nop')
        return ImportAll(node.id, node.relative)

    def copy_argument(self, argument: Argument) -> Argument:
        if False:
            print('Hello World!')
        arg = Argument(self.visit_var(argument.variable), argument.type_annotation, argument.initializer, argument.kind)
        arg.set_line(argument)
        return arg

    def visit_func_def(self, node: FuncDef) -> FuncDef:
        if False:
            i = 10
            return i + 15
        init = FuncMapInitializer(self)
        for stmt in node.body.body:
            stmt.accept(init)
        new = FuncDef(node.name, [self.copy_argument(arg) for arg in node.arguments], self.block(node.body), cast(Optional[FunctionLike], self.optional_type(node.type)))
        self.copy_function_attributes(new, node)
        new._fullname = node._fullname
        new.is_decorated = node.is_decorated
        new.is_conditional = node.is_conditional
        new.abstract_status = node.abstract_status
        new.is_static = node.is_static
        new.is_class = node.is_class
        new.is_property = node.is_property
        new.is_final = node.is_final
        new.original_def = node.original_def
        if node in self.func_placeholder_map:
            result = self.func_placeholder_map[node]
            replace_object_state(result, new)
            return result
        else:
            return new

    def visit_lambda_expr(self, node: LambdaExpr) -> LambdaExpr:
        if False:
            print('Hello World!')
        new = LambdaExpr([self.copy_argument(arg) for arg in node.arguments], self.block(node.body), cast(Optional[FunctionLike], self.optional_type(node.type)))
        self.copy_function_attributes(new, node)
        return new

    def copy_function_attributes(self, new: FuncItem, original: FuncItem) -> None:
        if False:
            while True:
                i = 10
        new.info = original.info
        new.min_args = original.min_args
        new.max_pos = original.max_pos
        new.is_overload = original.is_overload
        new.is_generator = original.is_generator
        new.is_coroutine = original.is_coroutine
        new.is_async_generator = original.is_async_generator
        new.is_awaitable_coroutine = original.is_awaitable_coroutine
        new.line = original.line

    def visit_overloaded_func_def(self, node: OverloadedFuncDef) -> OverloadedFuncDef:
        if False:
            i = 10
            return i + 15
        items = [cast(OverloadPart, item.accept(self)) for item in node.items]
        for (newitem, olditem) in zip(items, node.items):
            newitem.line = olditem.line
        new = OverloadedFuncDef(items)
        new._fullname = node._fullname
        new_type = self.optional_type(node.type)
        assert isinstance(new_type, ProperType)
        new.type = new_type
        new.info = node.info
        new.is_static = node.is_static
        new.is_class = node.is_class
        new.is_property = node.is_property
        new.is_final = node.is_final
        if node.impl:
            new.impl = cast(OverloadPart, node.impl.accept(self))
        return new

    def visit_class_def(self, node: ClassDef) -> ClassDef:
        if False:
            print('Hello World!')
        new = ClassDef(node.name, self.block(node.defs), node.type_vars, self.expressions(node.base_type_exprs), self.optional_expr(node.metaclass))
        new.fullname = node.fullname
        new.info = node.info
        new.decorators = [self.expr(decorator) for decorator in node.decorators]
        return new

    def visit_global_decl(self, node: GlobalDecl) -> GlobalDecl:
        if False:
            for i in range(10):
                print('nop')
        return GlobalDecl(node.names.copy())

    def visit_nonlocal_decl(self, node: NonlocalDecl) -> NonlocalDecl:
        if False:
            for i in range(10):
                print('nop')
        return NonlocalDecl(node.names.copy())

    def visit_block(self, node: Block) -> Block:
        if False:
            i = 10
            return i + 15
        return Block(self.statements(node.body))

    def visit_decorator(self, node: Decorator) -> Decorator:
        if False:
            print('Hello World!')
        func = self.visit_func_def(node.func)
        func.line = node.func.line
        new = Decorator(func, self.expressions(node.decorators), self.visit_var(node.var))
        new.is_overload = node.is_overload
        return new

    def visit_var(self, node: Var) -> Var:
        if False:
            return 10
        if node in self.var_map:
            return self.var_map[node]
        new = Var(node.name, self.optional_type(node.type))
        new.line = node.line
        new._fullname = node._fullname
        new.info = node.info
        new.is_self = node.is_self
        new.is_ready = node.is_ready
        new.is_initialized_in_class = node.is_initialized_in_class
        new.is_staticmethod = node.is_staticmethod
        new.is_classmethod = node.is_classmethod
        new.is_property = node.is_property
        new.is_final = node.is_final
        new.final_value = node.final_value
        new.final_unset_in_class = node.final_unset_in_class
        new.final_set_in_init = node.final_set_in_init
        new.set_line(node)
        self.var_map[node] = new
        return new

    def visit_expression_stmt(self, node: ExpressionStmt) -> ExpressionStmt:
        if False:
            return 10
        return ExpressionStmt(self.expr(node.expr))

    def visit_assignment_stmt(self, node: AssignmentStmt) -> AssignmentStmt:
        if False:
            return 10
        return self.duplicate_assignment(node)

    def duplicate_assignment(self, node: AssignmentStmt) -> AssignmentStmt:
        if False:
            while True:
                i = 10
        new = AssignmentStmt(self.expressions(node.lvalues), self.expr(node.rvalue), self.optional_type(node.unanalyzed_type))
        new.line = node.line
        new.is_final_def = node.is_final_def
        new.type = self.optional_type(node.type)
        return new

    def visit_operator_assignment_stmt(self, node: OperatorAssignmentStmt) -> OperatorAssignmentStmt:
        if False:
            i = 10
            return i + 15
        return OperatorAssignmentStmt(node.op, self.expr(node.lvalue), self.expr(node.rvalue))

    def visit_while_stmt(self, node: WhileStmt) -> WhileStmt:
        if False:
            while True:
                i = 10
        return WhileStmt(self.expr(node.expr), self.block(node.body), self.optional_block(node.else_body))

    def visit_for_stmt(self, node: ForStmt) -> ForStmt:
        if False:
            print('Hello World!')
        new = ForStmt(self.expr(node.index), self.expr(node.expr), self.block(node.body), self.optional_block(node.else_body), self.optional_type(node.unanalyzed_index_type))
        new.is_async = node.is_async
        new.index_type = self.optional_type(node.index_type)
        return new

    def visit_return_stmt(self, node: ReturnStmt) -> ReturnStmt:
        if False:
            while True:
                i = 10
        return ReturnStmt(self.optional_expr(node.expr))

    def visit_assert_stmt(self, node: AssertStmt) -> AssertStmt:
        if False:
            for i in range(10):
                print('nop')
        return AssertStmt(self.expr(node.expr), self.optional_expr(node.msg))

    def visit_del_stmt(self, node: DelStmt) -> DelStmt:
        if False:
            print('Hello World!')
        return DelStmt(self.expr(node.expr))

    def visit_if_stmt(self, node: IfStmt) -> IfStmt:
        if False:
            while True:
                i = 10
        return IfStmt(self.expressions(node.expr), self.blocks(node.body), self.optional_block(node.else_body))

    def visit_break_stmt(self, node: BreakStmt) -> BreakStmt:
        if False:
            print('Hello World!')
        return BreakStmt()

    def visit_continue_stmt(self, node: ContinueStmt) -> ContinueStmt:
        if False:
            print('Hello World!')
        return ContinueStmt()

    def visit_pass_stmt(self, node: PassStmt) -> PassStmt:
        if False:
            i = 10
            return i + 15
        return PassStmt()

    def visit_raise_stmt(self, node: RaiseStmt) -> RaiseStmt:
        if False:
            return 10
        return RaiseStmt(self.optional_expr(node.expr), self.optional_expr(node.from_expr))

    def visit_try_stmt(self, node: TryStmt) -> TryStmt:
        if False:
            print('Hello World!')
        new = TryStmt(self.block(node.body), self.optional_names(node.vars), self.optional_expressions(node.types), self.blocks(node.handlers), self.optional_block(node.else_body), self.optional_block(node.finally_body))
        new.is_star = node.is_star
        return new

    def visit_with_stmt(self, node: WithStmt) -> WithStmt:
        if False:
            i = 10
            return i + 15
        new = WithStmt(self.expressions(node.expr), self.optional_expressions(node.target), self.block(node.body), self.optional_type(node.unanalyzed_type))
        new.is_async = node.is_async
        new.analyzed_types = [self.type(typ) for typ in node.analyzed_types]
        return new

    def visit_as_pattern(self, p: AsPattern) -> AsPattern:
        if False:
            print('Hello World!')
        return AsPattern(pattern=self.pattern(p.pattern) if p.pattern is not None else None, name=self.duplicate_name(p.name) if p.name is not None else None)

    def visit_or_pattern(self, p: OrPattern) -> OrPattern:
        if False:
            return 10
        return OrPattern([self.pattern(pat) for pat in p.patterns])

    def visit_value_pattern(self, p: ValuePattern) -> ValuePattern:
        if False:
            return 10
        return ValuePattern(self.expr(p.expr))

    def visit_singleton_pattern(self, p: SingletonPattern) -> SingletonPattern:
        if False:
            for i in range(10):
                print('nop')
        return SingletonPattern(p.value)

    def visit_sequence_pattern(self, p: SequencePattern) -> SequencePattern:
        if False:
            i = 10
            return i + 15
        return SequencePattern([self.pattern(pat) for pat in p.patterns])

    def visit_starred_pattern(self, p: StarredPattern) -> StarredPattern:
        if False:
            i = 10
            return i + 15
        return StarredPattern(self.duplicate_name(p.capture) if p.capture is not None else None)

    def visit_mapping_pattern(self, p: MappingPattern) -> MappingPattern:
        if False:
            while True:
                i = 10
        return MappingPattern(keys=[self.expr(expr) for expr in p.keys], values=[self.pattern(pat) for pat in p.values], rest=self.duplicate_name(p.rest) if p.rest is not None else None)

    def visit_class_pattern(self, p: ClassPattern) -> ClassPattern:
        if False:
            for i in range(10):
                print('nop')
        class_ref = p.class_ref.accept(self)
        assert isinstance(class_ref, RefExpr)
        return ClassPattern(class_ref=class_ref, positionals=[self.pattern(pat) for pat in p.positionals], keyword_keys=list(p.keyword_keys), keyword_values=[self.pattern(pat) for pat in p.keyword_values])

    def visit_match_stmt(self, o: MatchStmt) -> MatchStmt:
        if False:
            while True:
                i = 10
        return MatchStmt(subject=self.expr(o.subject), patterns=[self.pattern(p) for p in o.patterns], guards=self.optional_expressions(o.guards), bodies=self.blocks(o.bodies))

    def visit_star_expr(self, node: StarExpr) -> StarExpr:
        if False:
            return 10
        return StarExpr(node.expr)

    def visit_int_expr(self, node: IntExpr) -> IntExpr:
        if False:
            i = 10
            return i + 15
        return IntExpr(node.value)

    def visit_str_expr(self, node: StrExpr) -> StrExpr:
        if False:
            for i in range(10):
                print('nop')
        return StrExpr(node.value)

    def visit_bytes_expr(self, node: BytesExpr) -> BytesExpr:
        if False:
            i = 10
            return i + 15
        return BytesExpr(node.value)

    def visit_float_expr(self, node: FloatExpr) -> FloatExpr:
        if False:
            return 10
        return FloatExpr(node.value)

    def visit_complex_expr(self, node: ComplexExpr) -> ComplexExpr:
        if False:
            return 10
        return ComplexExpr(node.value)

    def visit_ellipsis(self, node: EllipsisExpr) -> EllipsisExpr:
        if False:
            print('Hello World!')
        return EllipsisExpr()

    def visit_name_expr(self, node: NameExpr) -> NameExpr:
        if False:
            for i in range(10):
                print('nop')
        return self.duplicate_name(node)

    def duplicate_name(self, node: NameExpr) -> NameExpr:
        if False:
            i = 10
            return i + 15
        new = NameExpr(node.name)
        self.copy_ref(new, node)
        new.is_special_form = node.is_special_form
        return new

    def visit_member_expr(self, node: MemberExpr) -> MemberExpr:
        if False:
            i = 10
            return i + 15
        member = MemberExpr(self.expr(node.expr), node.name)
        if node.def_var:
            member.def_var = node.def_var
        self.copy_ref(member, node)
        return member

    def copy_ref(self, new: RefExpr, original: RefExpr) -> None:
        if False:
            return 10
        new.kind = original.kind
        new.fullname = original.fullname
        target = original.node
        if isinstance(target, Var):
            if original.kind != GDEF:
                target = self.visit_var(target)
        elif isinstance(target, Decorator):
            target = self.visit_var(target.var)
        elif isinstance(target, FuncDef):
            target = self.func_placeholder_map.get(target, target)
        new.node = target
        new.is_new_def = original.is_new_def
        new.is_inferred_def = original.is_inferred_def

    def visit_yield_from_expr(self, node: YieldFromExpr) -> YieldFromExpr:
        if False:
            print('Hello World!')
        return YieldFromExpr(self.expr(node.expr))

    def visit_yield_expr(self, node: YieldExpr) -> YieldExpr:
        if False:
            print('Hello World!')
        return YieldExpr(self.optional_expr(node.expr))

    def visit_await_expr(self, node: AwaitExpr) -> AwaitExpr:
        if False:
            i = 10
            return i + 15
        return AwaitExpr(self.expr(node.expr))

    def visit_call_expr(self, node: CallExpr) -> CallExpr:
        if False:
            print('Hello World!')
        return CallExpr(self.expr(node.callee), self.expressions(node.args), node.arg_kinds.copy(), node.arg_names.copy(), self.optional_expr(node.analyzed))

    def visit_op_expr(self, node: OpExpr) -> OpExpr:
        if False:
            i = 10
            return i + 15
        new = OpExpr(node.op, self.expr(node.left), self.expr(node.right), cast(Optional[TypeAliasExpr], self.optional_expr(node.analyzed)))
        new.method_type = self.optional_type(node.method_type)
        return new

    def visit_comparison_expr(self, node: ComparisonExpr) -> ComparisonExpr:
        if False:
            for i in range(10):
                print('nop')
        new = ComparisonExpr(node.operators, self.expressions(node.operands))
        new.method_types = [self.optional_type(t) for t in node.method_types]
        return new

    def visit_cast_expr(self, node: CastExpr) -> CastExpr:
        if False:
            return 10
        return CastExpr(self.expr(node.expr), self.type(node.type))

    def visit_assert_type_expr(self, node: AssertTypeExpr) -> AssertTypeExpr:
        if False:
            while True:
                i = 10
        return AssertTypeExpr(self.expr(node.expr), self.type(node.type))

    def visit_reveal_expr(self, node: RevealExpr) -> RevealExpr:
        if False:
            i = 10
            return i + 15
        if node.kind == REVEAL_TYPE:
            assert node.expr is not None
            return RevealExpr(kind=REVEAL_TYPE, expr=self.expr(node.expr))
        else:
            return node

    def visit_super_expr(self, node: SuperExpr) -> SuperExpr:
        if False:
            for i in range(10):
                print('nop')
        call = self.expr(node.call)
        assert isinstance(call, CallExpr)
        new = SuperExpr(node.name, call)
        new.info = node.info
        return new

    def visit_assignment_expr(self, node: AssignmentExpr) -> AssignmentExpr:
        if False:
            while True:
                i = 10
        return AssignmentExpr(self.expr(node.target), self.expr(node.value))

    def visit_unary_expr(self, node: UnaryExpr) -> UnaryExpr:
        if False:
            return 10
        new = UnaryExpr(node.op, self.expr(node.expr))
        new.method_type = self.optional_type(node.method_type)
        return new

    def visit_list_expr(self, node: ListExpr) -> ListExpr:
        if False:
            for i in range(10):
                print('nop')
        return ListExpr(self.expressions(node.items))

    def visit_dict_expr(self, node: DictExpr) -> DictExpr:
        if False:
            for i in range(10):
                print('nop')
        return DictExpr([(self.expr(key) if key else None, self.expr(value)) for (key, value) in node.items])

    def visit_tuple_expr(self, node: TupleExpr) -> TupleExpr:
        if False:
            while True:
                i = 10
        return TupleExpr(self.expressions(node.items))

    def visit_set_expr(self, node: SetExpr) -> SetExpr:
        if False:
            return 10
        return SetExpr(self.expressions(node.items))

    def visit_index_expr(self, node: IndexExpr) -> IndexExpr:
        if False:
            for i in range(10):
                print('nop')
        new = IndexExpr(self.expr(node.base), self.expr(node.index))
        if node.method_type:
            new.method_type = self.type(node.method_type)
        if node.analyzed:
            if isinstance(node.analyzed, TypeApplication):
                new.analyzed = self.visit_type_application(node.analyzed)
            else:
                new.analyzed = self.visit_type_alias_expr(node.analyzed)
            new.analyzed.set_line(node.analyzed)
        return new

    def visit_type_application(self, node: TypeApplication) -> TypeApplication:
        if False:
            for i in range(10):
                print('nop')
        return TypeApplication(self.expr(node.expr), self.types(node.types))

    def visit_list_comprehension(self, node: ListComprehension) -> ListComprehension:
        if False:
            for i in range(10):
                print('nop')
        generator = self.duplicate_generator(node.generator)
        generator.set_line(node.generator)
        return ListComprehension(generator)

    def visit_set_comprehension(self, node: SetComprehension) -> SetComprehension:
        if False:
            while True:
                i = 10
        generator = self.duplicate_generator(node.generator)
        generator.set_line(node.generator)
        return SetComprehension(generator)

    def visit_dictionary_comprehension(self, node: DictionaryComprehension) -> DictionaryComprehension:
        if False:
            for i in range(10):
                print('nop')
        return DictionaryComprehension(self.expr(node.key), self.expr(node.value), [self.expr(index) for index in node.indices], [self.expr(s) for s in node.sequences], [[self.expr(cond) for cond in conditions] for conditions in node.condlists], node.is_async)

    def visit_generator_expr(self, node: GeneratorExpr) -> GeneratorExpr:
        if False:
            while True:
                i = 10
        return self.duplicate_generator(node)

    def duplicate_generator(self, node: GeneratorExpr) -> GeneratorExpr:
        if False:
            while True:
                i = 10
        return GeneratorExpr(self.expr(node.left_expr), [self.expr(index) for index in node.indices], [self.expr(s) for s in node.sequences], [[self.expr(cond) for cond in conditions] for conditions in node.condlists], node.is_async)

    def visit_slice_expr(self, node: SliceExpr) -> SliceExpr:
        if False:
            i = 10
            return i + 15
        return SliceExpr(self.optional_expr(node.begin_index), self.optional_expr(node.end_index), self.optional_expr(node.stride))

    def visit_conditional_expr(self, node: ConditionalExpr) -> ConditionalExpr:
        if False:
            while True:
                i = 10
        return ConditionalExpr(self.expr(node.cond), self.expr(node.if_expr), self.expr(node.else_expr))

    def visit_type_var_expr(self, node: TypeVarExpr) -> TypeVarExpr:
        if False:
            for i in range(10):
                print('nop')
        return TypeVarExpr(node.name, node.fullname, self.types(node.values), self.type(node.upper_bound), self.type(node.default), variance=node.variance)

    def visit_paramspec_expr(self, node: ParamSpecExpr) -> ParamSpecExpr:
        if False:
            while True:
                i = 10
        return ParamSpecExpr(node.name, node.fullname, self.type(node.upper_bound), self.type(node.default), variance=node.variance)

    def visit_type_var_tuple_expr(self, node: TypeVarTupleExpr) -> TypeVarTupleExpr:
        if False:
            for i in range(10):
                print('nop')
        return TypeVarTupleExpr(node.name, node.fullname, self.type(node.upper_bound), node.tuple_fallback, self.type(node.default), variance=node.variance)

    def visit_type_alias_expr(self, node: TypeAliasExpr) -> TypeAliasExpr:
        if False:
            return 10
        return TypeAliasExpr(node.node)

    def visit_newtype_expr(self, node: NewTypeExpr) -> NewTypeExpr:
        if False:
            while True:
                i = 10
        res = NewTypeExpr(node.name, node.old_type, line=node.line, column=node.column)
        res.info = node.info
        return res

    def visit_namedtuple_expr(self, node: NamedTupleExpr) -> NamedTupleExpr:
        if False:
            i = 10
            return i + 15
        return NamedTupleExpr(node.info)

    def visit_enum_call_expr(self, node: EnumCallExpr) -> EnumCallExpr:
        if False:
            print('Hello World!')
        return EnumCallExpr(node.info, node.items, node.values)

    def visit_typeddict_expr(self, node: TypedDictExpr) -> Node:
        if False:
            return 10
        return TypedDictExpr(node.info)

    def visit__promote_expr(self, node: PromoteExpr) -> PromoteExpr:
        if False:
            return 10
        return PromoteExpr(node.type)

    def visit_temp_node(self, node: TempNode) -> TempNode:
        if False:
            while True:
                i = 10
        return TempNode(self.type(node.type))

    def node(self, node: Node) -> Node:
        if False:
            i = 10
            return i + 15
        new = node.accept(self)
        new.set_line(node)
        return new

    def mypyfile(self, node: MypyFile) -> MypyFile:
        if False:
            while True:
                i = 10
        new = node.accept(self)
        assert isinstance(new, MypyFile)
        new.set_line(node)
        return new

    def expr(self, expr: Expression) -> Expression:
        if False:
            return 10
        new = expr.accept(self)
        assert isinstance(new, Expression)
        new.set_line(expr)
        return new

    def stmt(self, stmt: Statement) -> Statement:
        if False:
            for i in range(10):
                print('nop')
        new = stmt.accept(self)
        assert isinstance(new, Statement)
        new.set_line(stmt)
        return new

    def pattern(self, pattern: Pattern) -> Pattern:
        if False:
            i = 10
            return i + 15
        new = pattern.accept(self)
        assert isinstance(new, Pattern)
        new.set_line(pattern)
        return new

    def optional_expr(self, expr: Expression | None) -> Expression | None:
        if False:
            print('Hello World!')
        if expr:
            return self.expr(expr)
        else:
            return None

    def block(self, block: Block) -> Block:
        if False:
            return 10
        new = self.visit_block(block)
        new.line = block.line
        return new

    def optional_block(self, block: Block | None) -> Block | None:
        if False:
            i = 10
            return i + 15
        if block:
            return self.block(block)
        else:
            return None

    def statements(self, statements: list[Statement]) -> list[Statement]:
        if False:
            for i in range(10):
                print('nop')
        return [self.stmt(stmt) for stmt in statements]

    def expressions(self, expressions: list[Expression]) -> list[Expression]:
        if False:
            while True:
                i = 10
        return [self.expr(expr) for expr in expressions]

    def optional_expressions(self, expressions: Iterable[Expression | None]) -> list[Expression | None]:
        if False:
            for i in range(10):
                print('nop')
        return [self.optional_expr(expr) for expr in expressions]

    def blocks(self, blocks: list[Block]) -> list[Block]:
        if False:
            while True:
                i = 10
        return [self.block(block) for block in blocks]

    def names(self, names: list[NameExpr]) -> list[NameExpr]:
        if False:
            return 10
        return [self.duplicate_name(name) for name in names]

    def optional_names(self, names: Iterable[NameExpr | None]) -> list[NameExpr | None]:
        if False:
            for i in range(10):
                print('nop')
        result: list[NameExpr | None] = []
        for name in names:
            if name:
                result.append(self.duplicate_name(name))
            else:
                result.append(None)
        return result

    def type(self, type: Type) -> Type:
        if False:
            return 10
        return type

    def optional_type(self, type: Type | None) -> Type | None:
        if False:
            while True:
                i = 10
        if type:
            return self.type(type)
        else:
            return None

    def types(self, types: list[Type]) -> list[Type]:
        if False:
            i = 10
            return i + 15
        return [self.type(type) for type in types]

class FuncMapInitializer(TraverserVisitor):
    """This traverser creates mappings from nested FuncDefs to placeholder FuncDefs.

    The placeholders will later be replaced with transformed nodes.
    """

    def __init__(self, transformer: TransformVisitor) -> None:
        if False:
            i = 10
            return i + 15
        self.transformer = transformer

    def visit_func_def(self, node: FuncDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        if node not in self.transformer.func_placeholder_map:
            self.transformer.func_placeholder_map[node] = FuncDef(node.name, node.arguments, node.body, None)
        super().visit_func_def(node)