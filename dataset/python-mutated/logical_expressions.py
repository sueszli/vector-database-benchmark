"""Converter for logical expressions, e.g. `a and b -> tf.logical_and(a, b)`."""
import gast
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import templates
SAFE_BOOLEAN_OPERAND = 'SAFE_BOOLEAN_OPERAND'
LOGICAL_OPERATORS = {gast.And: 'ag__.and_', gast.Not: 'ag__.not_', gast.Or: 'ag__.or_'}
EQUALITY_OPERATORS = {gast.Eq: 'ag__.eq', gast.NotEq: 'ag__.not_eq'}

class LogicalExpressionTransformer(converter.Base):
    """Converts logical expressions to corresponding TF calls."""

    def _overload_of(self, operator):
        if False:
            return 10
        op_type = type(operator)
        if op_type in LOGICAL_OPERATORS:
            return LOGICAL_OPERATORS[op_type]
        if self.ctx.user.options.uses(converter.Feature.EQUALITY_OPERATORS):
            if op_type in EQUALITY_OPERATORS:
                return EQUALITY_OPERATORS[op_type]
        return None

    def _as_lambda(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return templates.replace_as_expression('lambda: expr', expr=expr)

    def _as_binary_function(self, func_name, arg1, arg2):
        if False:
            for i in range(10):
                print('nop')
        return templates.replace_as_expression('func_name(arg1, arg2)', func_name=parser.parse_expression(func_name), arg1=arg1, arg2=arg2)

    def _as_binary_operation(self, op, arg1, arg2):
        if False:
            i = 10
            return i + 15
        template = templates.replace_as_expression('arg1 is arg2', arg1=arg1, arg2=arg2)
        template.ops[0] = op
        return template

    def _as_unary_function(self, func_name, arg):
        if False:
            print('Hello World!')
        return templates.replace_as_expression('func_name(arg)', func_name=parser.parse_expression(func_name), arg=arg)

    def _process_binop(self, op, left, right):
        if False:
            print('Hello World!')
        overload = self._overload_of(op)
        if overload is None:
            return self._as_binary_operation(op, left, right)
        return self._as_binary_function(overload, left, right)

    def visit_Compare(self, node):
        if False:
            i = 10
            return i + 15
        node = self.generic_visit(node)
        ops_and_comps = list(zip(node.ops, node.comparators))
        left = node.left
        op_tree = None
        while ops_and_comps:
            (op, right) = ops_and_comps.pop(0)
            binary_comparison = self._process_binop(op, left, right)
            if op_tree is not None:
                op_tree = self._as_binary_function('ag__.and_', self._as_lambda(op_tree), self._as_lambda(binary_comparison))
            else:
                op_tree = binary_comparison
            left = right
        assert op_tree is not None
        return op_tree

    def visit_UnaryOp(self, node):
        if False:
            i = 10
            return i + 15
        node = self.generic_visit(node)
        overload = self._overload_of(node.op)
        if overload is None:
            return node
        return self._as_unary_function(overload, node.operand)

    def visit_BoolOp(self, node):
        if False:
            while True:
                i = 10
        node = self.generic_visit(node)
        node_values = node.values
        right = node.values.pop()
        while node_values:
            left = node_values.pop()
            right = self._as_binary_function(self._overload_of(node.op), self._as_lambda(left), self._as_lambda(right))
        return right

def transform(node, ctx):
    if False:
        i = 10
        return i + 15
    transformer = LogicalExpressionTransformer(ctx)
    return transformer.visit(node)