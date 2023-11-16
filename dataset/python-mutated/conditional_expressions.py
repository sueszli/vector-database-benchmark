"""Converts the ternary conditional operator."""
import gast
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import templates

class ConditionalExpressionTransformer(converter.Base):
    """Converts conditional expressions to functional form."""

    def visit_IfExp(self, node):
        if False:
            return 10
        template = '\n        ag__.if_exp(\n            test,\n            lambda: true_expr,\n            lambda: false_expr,\n            expr_repr)\n    '
        expr_repr = parser.unparse(node.test, include_encoding_marker=False).strip()
        return templates.replace_as_expression(template, test=node.test, true_expr=node.body, false_expr=node.orelse, expr_repr=gast.Constant(expr_repr, kind=None))

def transform(node, ctx):
    if False:
        while True:
            i = 10
    node = ConditionalExpressionTransformer(ctx).visit(node)
    return node