from __future__ import annotations
from mypy.nodes import AssertTypeExpr, AssignmentStmt, CastExpr, ClassDef, ForStmt, FuncItem, NamedTupleExpr, NewTypeExpr, PromoteExpr, TypeAliasExpr, TypeApplication, TypedDictExpr, TypeVarExpr, Var, WithStmt
from mypy.traverser import TraverserVisitor
from mypy.types import Type
from mypy.typetraverser import TypeTraverserVisitor

class MixedTraverserVisitor(TraverserVisitor, TypeTraverserVisitor):
    """Recursive traversal of both Node and Type objects."""

    def __init__(self) -> None:
        if False:
            return 10
        self.in_type_alias_expr = False

    def visit_var(self, var: Var) -> None:
        if False:
            print('Hello World!')
        self.visit_optional_type(var.type)

    def visit_func(self, o: FuncItem) -> None:
        if False:
            print('Hello World!')
        super().visit_func(o)
        self.visit_optional_type(o.type)

    def visit_class_def(self, o: ClassDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().visit_class_def(o)
        info = o.info
        if info:
            for base in info.bases:
                base.accept(self)

    def visit_type_alias_expr(self, o: TypeAliasExpr) -> None:
        if False:
            print('Hello World!')
        super().visit_type_alias_expr(o)
        self.in_type_alias_expr = True
        o.node.target.accept(self)
        self.in_type_alias_expr = False

    def visit_type_var_expr(self, o: TypeVarExpr) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_type_var_expr(o)
        o.upper_bound.accept(self)
        for value in o.values:
            value.accept(self)

    def visit_typeddict_expr(self, o: TypedDictExpr) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_typeddict_expr(o)
        self.visit_optional_type(o.info.typeddict_type)

    def visit_namedtuple_expr(self, o: NamedTupleExpr) -> None:
        if False:
            while True:
                i = 10
        super().visit_namedtuple_expr(o)
        assert o.info.tuple_type
        o.info.tuple_type.accept(self)

    def visit__promote_expr(self, o: PromoteExpr) -> None:
        if False:
            print('Hello World!')
        super().visit__promote_expr(o)
        o.type.accept(self)

    def visit_newtype_expr(self, o: NewTypeExpr) -> None:
        if False:
            while True:
                i = 10
        super().visit_newtype_expr(o)
        self.visit_optional_type(o.old_type)

    def visit_assignment_stmt(self, o: AssignmentStmt) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().visit_assignment_stmt(o)
        self.visit_optional_type(o.type)

    def visit_for_stmt(self, o: ForStmt) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_for_stmt(o)
        self.visit_optional_type(o.index_type)

    def visit_with_stmt(self, o: WithStmt) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_with_stmt(o)
        for typ in o.analyzed_types:
            typ.accept(self)

    def visit_cast_expr(self, o: CastExpr) -> None:
        if False:
            print('Hello World!')
        super().visit_cast_expr(o)
        o.type.accept(self)

    def visit_assert_type_expr(self, o: AssertTypeExpr) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_assert_type_expr(o)
        o.type.accept(self)

    def visit_type_application(self, o: TypeApplication) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().visit_type_application(o)
        for t in o.types:
            t.accept(self)

    def visit_optional_type(self, t: Type | None) -> None:
        if False:
            print('Hello World!')
        if t:
            t.accept(self)