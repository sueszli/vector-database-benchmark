"""

Evaluate an expression.

Used by stubtest; in a separate file because things break if we don't
put it in a mypyc-compiled file.

"""
import ast
from typing import Final
import mypy.nodes
from mypy.visitor import ExpressionVisitor
UNKNOWN = object()

class _NodeEvaluator(ExpressionVisitor[object]):

    def visit_int_expr(self, o: mypy.nodes.IntExpr) -> int:
        if False:
            while True:
                i = 10
        return o.value

    def visit_str_expr(self, o: mypy.nodes.StrExpr) -> str:
        if False:
            for i in range(10):
                print('nop')
        return o.value

    def visit_bytes_expr(self, o: mypy.nodes.BytesExpr) -> object:
        if False:
            print('Hello World!')
        try:
            return ast.literal_eval(f"b'{o.value}'")
        except SyntaxError:
            return ast.literal_eval(f'b"{o.value}"')

    def visit_float_expr(self, o: mypy.nodes.FloatExpr) -> float:
        if False:
            print('Hello World!')
        return o.value

    def visit_complex_expr(self, o: mypy.nodes.ComplexExpr) -> object:
        if False:
            i = 10
            return i + 15
        return o.value

    def visit_ellipsis(self, o: mypy.nodes.EllipsisExpr) -> object:
        if False:
            print('Hello World!')
        return Ellipsis

    def visit_star_expr(self, o: mypy.nodes.StarExpr) -> object:
        if False:
            return 10
        return UNKNOWN

    def visit_name_expr(self, o: mypy.nodes.NameExpr) -> object:
        if False:
            print('Hello World!')
        if o.name == 'True':
            return True
        elif o.name == 'False':
            return False
        elif o.name == 'None':
            return None
        return UNKNOWN

    def visit_member_expr(self, o: mypy.nodes.MemberExpr) -> object:
        if False:
            i = 10
            return i + 15
        return UNKNOWN

    def visit_yield_from_expr(self, o: mypy.nodes.YieldFromExpr) -> object:
        if False:
            print('Hello World!')
        return UNKNOWN

    def visit_yield_expr(self, o: mypy.nodes.YieldExpr) -> object:
        if False:
            i = 10
            return i + 15
        return UNKNOWN

    def visit_call_expr(self, o: mypy.nodes.CallExpr) -> object:
        if False:
            print('Hello World!')
        return UNKNOWN

    def visit_op_expr(self, o: mypy.nodes.OpExpr) -> object:
        if False:
            print('Hello World!')
        return UNKNOWN

    def visit_comparison_expr(self, o: mypy.nodes.ComparisonExpr) -> object:
        if False:
            while True:
                i = 10
        return UNKNOWN

    def visit_cast_expr(self, o: mypy.nodes.CastExpr) -> object:
        if False:
            while True:
                i = 10
        return o.expr.accept(self)

    def visit_assert_type_expr(self, o: mypy.nodes.AssertTypeExpr) -> object:
        if False:
            while True:
                i = 10
        return o.expr.accept(self)

    def visit_reveal_expr(self, o: mypy.nodes.RevealExpr) -> object:
        if False:
            for i in range(10):
                print('nop')
        return UNKNOWN

    def visit_super_expr(self, o: mypy.nodes.SuperExpr) -> object:
        if False:
            return 10
        return UNKNOWN

    def visit_unary_expr(self, o: mypy.nodes.UnaryExpr) -> object:
        if False:
            i = 10
            return i + 15
        operand = o.expr.accept(self)
        if operand is UNKNOWN:
            return UNKNOWN
        if o.op == '-':
            if isinstance(operand, (int, float, complex)):
                return -operand
        elif o.op == '+':
            if isinstance(operand, (int, float, complex)):
                return +operand
        elif o.op == '~':
            if isinstance(operand, int):
                return ~operand
        elif o.op == 'not':
            if isinstance(operand, (bool, int, float, str, bytes)):
                return not operand
        return UNKNOWN

    def visit_assignment_expr(self, o: mypy.nodes.AssignmentExpr) -> object:
        if False:
            for i in range(10):
                print('nop')
        return o.value.accept(self)

    def visit_list_expr(self, o: mypy.nodes.ListExpr) -> object:
        if False:
            print('Hello World!')
        items = [item.accept(self) for item in o.items]
        if all((item is not UNKNOWN for item in items)):
            return items
        return UNKNOWN

    def visit_dict_expr(self, o: mypy.nodes.DictExpr) -> object:
        if False:
            for i in range(10):
                print('nop')
        items = [(UNKNOWN if key is None else key.accept(self), value.accept(self)) for (key, value) in o.items]
        if all((key is not UNKNOWN and value is not None for (key, value) in items)):
            return dict(items)
        return UNKNOWN

    def visit_tuple_expr(self, o: mypy.nodes.TupleExpr) -> object:
        if False:
            print('Hello World!')
        items = [item.accept(self) for item in o.items]
        if all((item is not UNKNOWN for item in items)):
            return tuple(items)
        return UNKNOWN

    def visit_set_expr(self, o: mypy.nodes.SetExpr) -> object:
        if False:
            while True:
                i = 10
        items = [item.accept(self) for item in o.items]
        if all((item is not UNKNOWN for item in items)):
            return set(items)
        return UNKNOWN

    def visit_index_expr(self, o: mypy.nodes.IndexExpr) -> object:
        if False:
            while True:
                i = 10
        return UNKNOWN

    def visit_type_application(self, o: mypy.nodes.TypeApplication) -> object:
        if False:
            for i in range(10):
                print('nop')
        return UNKNOWN

    def visit_lambda_expr(self, o: mypy.nodes.LambdaExpr) -> object:
        if False:
            for i in range(10):
                print('nop')
        return UNKNOWN

    def visit_list_comprehension(self, o: mypy.nodes.ListComprehension) -> object:
        if False:
            while True:
                i = 10
        return UNKNOWN

    def visit_set_comprehension(self, o: mypy.nodes.SetComprehension) -> object:
        if False:
            for i in range(10):
                print('nop')
        return UNKNOWN

    def visit_dictionary_comprehension(self, o: mypy.nodes.DictionaryComprehension) -> object:
        if False:
            return 10
        return UNKNOWN

    def visit_generator_expr(self, o: mypy.nodes.GeneratorExpr) -> object:
        if False:
            i = 10
            return i + 15
        return UNKNOWN

    def visit_slice_expr(self, o: mypy.nodes.SliceExpr) -> object:
        if False:
            while True:
                i = 10
        return UNKNOWN

    def visit_conditional_expr(self, o: mypy.nodes.ConditionalExpr) -> object:
        if False:
            i = 10
            return i + 15
        return UNKNOWN

    def visit_type_var_expr(self, o: mypy.nodes.TypeVarExpr) -> object:
        if False:
            return 10
        return UNKNOWN

    def visit_paramspec_expr(self, o: mypy.nodes.ParamSpecExpr) -> object:
        if False:
            while True:
                i = 10
        return UNKNOWN

    def visit_type_var_tuple_expr(self, o: mypy.nodes.TypeVarTupleExpr) -> object:
        if False:
            print('Hello World!')
        return UNKNOWN

    def visit_type_alias_expr(self, o: mypy.nodes.TypeAliasExpr) -> object:
        if False:
            while True:
                i = 10
        return UNKNOWN

    def visit_namedtuple_expr(self, o: mypy.nodes.NamedTupleExpr) -> object:
        if False:
            while True:
                i = 10
        return UNKNOWN

    def visit_enum_call_expr(self, o: mypy.nodes.EnumCallExpr) -> object:
        if False:
            return 10
        return UNKNOWN

    def visit_typeddict_expr(self, o: mypy.nodes.TypedDictExpr) -> object:
        if False:
            for i in range(10):
                print('nop')
        return UNKNOWN

    def visit_newtype_expr(self, o: mypy.nodes.NewTypeExpr) -> object:
        if False:
            for i in range(10):
                print('nop')
        return UNKNOWN

    def visit__promote_expr(self, o: mypy.nodes.PromoteExpr) -> object:
        if False:
            print('Hello World!')
        return UNKNOWN

    def visit_await_expr(self, o: mypy.nodes.AwaitExpr) -> object:
        if False:
            return 10
        return UNKNOWN

    def visit_temp_node(self, o: mypy.nodes.TempNode) -> object:
        if False:
            print('Hello World!')
        return UNKNOWN
_evaluator: Final = _NodeEvaluator()

def evaluate_expression(expr: mypy.nodes.Expression) -> object:
    if False:
        for i in range(10):
            print('nop')
    'Evaluate an expression at runtime.\n\n    Return the result of the expression, or UNKNOWN if the expression cannot be\n    evaluated.\n    '
    return expr.accept(_evaluator)