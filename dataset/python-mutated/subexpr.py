"""Find all subexpressions of an AST node."""
from __future__ import annotations
from mypy.nodes import AssertTypeExpr, AssignmentExpr, AwaitExpr, CallExpr, CastExpr, ComparisonExpr, ConditionalExpr, DictExpr, DictionaryComprehension, Expression, GeneratorExpr, IndexExpr, LambdaExpr, ListComprehension, ListExpr, MemberExpr, Node, OpExpr, RevealExpr, SetComprehension, SetExpr, SliceExpr, StarExpr, TupleExpr, TypeApplication, UnaryExpr, YieldExpr, YieldFromExpr
from mypy.traverser import TraverserVisitor

def get_subexpressions(node: Node) -> list[Expression]:
    if False:
        i = 10
        return i + 15
    visitor = SubexpressionFinder()
    node.accept(visitor)
    return visitor.expressions

class SubexpressionFinder(TraverserVisitor):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.expressions: list[Expression] = []

    def visit_int_expr(self, o: Expression) -> None:
        if False:
            print('Hello World!')
        self.add(o)

    def visit_name_expr(self, o: Expression) -> None:
        if False:
            while True:
                i = 10
        self.add(o)

    def visit_float_expr(self, o: Expression) -> None:
        if False:
            i = 10
            return i + 15
        self.add(o)

    def visit_str_expr(self, o: Expression) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add(o)

    def visit_bytes_expr(self, o: Expression) -> None:
        if False:
            while True:
                i = 10
        self.add(o)

    def visit_unicode_expr(self, o: Expression) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add(o)

    def visit_complex_expr(self, o: Expression) -> None:
        if False:
            while True:
                i = 10
        self.add(o)

    def visit_ellipsis(self, o: Expression) -> None:
        if False:
            i = 10
            return i + 15
        self.add(o)

    def visit_super_expr(self, o: Expression) -> None:
        if False:
            print('Hello World!')
        self.add(o)

    def visit_type_var_expr(self, o: Expression) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add(o)

    def visit_type_alias_expr(self, o: Expression) -> None:
        if False:
            i = 10
            return i + 15
        self.add(o)

    def visit_namedtuple_expr(self, o: Expression) -> None:
        if False:
            print('Hello World!')
        self.add(o)

    def visit_typeddict_expr(self, o: Expression) -> None:
        if False:
            return 10
        self.add(o)

    def visit__promote_expr(self, o: Expression) -> None:
        if False:
            while True:
                i = 10
        self.add(o)

    def visit_newtype_expr(self, o: Expression) -> None:
        if False:
            while True:
                i = 10
        self.add(o)

    def visit_member_expr(self, e: MemberExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add(e)
        super().visit_member_expr(e)

    def visit_yield_from_expr(self, e: YieldFromExpr) -> None:
        if False:
            return 10
        self.add(e)
        super().visit_yield_from_expr(e)

    def visit_yield_expr(self, e: YieldExpr) -> None:
        if False:
            while True:
                i = 10
        self.add(e)
        super().visit_yield_expr(e)

    def visit_call_expr(self, e: CallExpr) -> None:
        if False:
            while True:
                i = 10
        self.add(e)
        super().visit_call_expr(e)

    def visit_op_expr(self, e: OpExpr) -> None:
        if False:
            print('Hello World!')
        self.add(e)
        super().visit_op_expr(e)

    def visit_comparison_expr(self, e: ComparisonExpr) -> None:
        if False:
            i = 10
            return i + 15
        self.add(e)
        super().visit_comparison_expr(e)

    def visit_slice_expr(self, e: SliceExpr) -> None:
        if False:
            while True:
                i = 10
        self.add(e)
        super().visit_slice_expr(e)

    def visit_cast_expr(self, e: CastExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add(e)
        super().visit_cast_expr(e)

    def visit_assert_type_expr(self, e: AssertTypeExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add(e)
        super().visit_assert_type_expr(e)

    def visit_reveal_expr(self, e: RevealExpr) -> None:
        if False:
            while True:
                i = 10
        self.add(e)
        super().visit_reveal_expr(e)

    def visit_assignment_expr(self, e: AssignmentExpr) -> None:
        if False:
            print('Hello World!')
        self.add(e)
        super().visit_assignment_expr(e)

    def visit_unary_expr(self, e: UnaryExpr) -> None:
        if False:
            return 10
        self.add(e)
        super().visit_unary_expr(e)

    def visit_list_expr(self, e: ListExpr) -> None:
        if False:
            print('Hello World!')
        self.add(e)
        super().visit_list_expr(e)

    def visit_tuple_expr(self, e: TupleExpr) -> None:
        if False:
            while True:
                i = 10
        self.add(e)
        super().visit_tuple_expr(e)

    def visit_dict_expr(self, e: DictExpr) -> None:
        if False:
            i = 10
            return i + 15
        self.add(e)
        super().visit_dict_expr(e)

    def visit_set_expr(self, e: SetExpr) -> None:
        if False:
            print('Hello World!')
        self.add(e)
        super().visit_set_expr(e)

    def visit_index_expr(self, e: IndexExpr) -> None:
        if False:
            while True:
                i = 10
        self.add(e)
        super().visit_index_expr(e)

    def visit_generator_expr(self, e: GeneratorExpr) -> None:
        if False:
            return 10
        self.add(e)
        super().visit_generator_expr(e)

    def visit_dictionary_comprehension(self, e: DictionaryComprehension) -> None:
        if False:
            while True:
                i = 10
        self.add(e)
        super().visit_dictionary_comprehension(e)

    def visit_list_comprehension(self, e: ListComprehension) -> None:
        if False:
            print('Hello World!')
        self.add(e)
        super().visit_list_comprehension(e)

    def visit_set_comprehension(self, e: SetComprehension) -> None:
        if False:
            return 10
        self.add(e)
        super().visit_set_comprehension(e)

    def visit_conditional_expr(self, e: ConditionalExpr) -> None:
        if False:
            i = 10
            return i + 15
        self.add(e)
        super().visit_conditional_expr(e)

    def visit_type_application(self, e: TypeApplication) -> None:
        if False:
            return 10
        self.add(e)
        super().visit_type_application(e)

    def visit_lambda_expr(self, e: LambdaExpr) -> None:
        if False:
            print('Hello World!')
        self.add(e)
        super().visit_lambda_expr(e)

    def visit_star_expr(self, e: StarExpr) -> None:
        if False:
            i = 10
            return i + 15
        self.add(e)
        super().visit_star_expr(e)

    def visit_await_expr(self, e: AwaitExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add(e)
        super().visit_await_expr(e)

    def add(self, e: Expression) -> None:
        if False:
            return 10
        self.expressions.append(e)