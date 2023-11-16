"""Lowers list comprehensions into for and if statements.

Example:

  result = [x * x for x in xs]

becomes

  result = []
  for x in xs:
    elt = x * x
    result.append(elt)
"""
import gast
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.pyct import templates

class ListCompTransformer(converter.Base):
    """Lowers list comprehensions into standard control flow."""

    def visit_Assign(self, node):
        if False:
            while True:
                i = 10
        if not isinstance(node.value, gast.ListComp):
            return self.generic_visit(node)
        if len(node.targets) > 1:
            raise NotImplementedError('multiple assignments')
        (target,) = node.targets
        list_comp_node = node.value
        template = '\n      target = []\n    '
        initialization = templates.replace(template, target=target)
        template = '\n      target.append(elt)\n    '
        body = templates.replace(template, target=target, elt=list_comp_node.elt)
        for gen in reversed(list_comp_node.generators):
            for gen_if in reversed(gen.ifs):
                template = '\n          if test:\n            body\n        '
                body = templates.replace(template, test=gen_if, body=body)
            template = '\n        for target in iter_:\n          body\n      '
            body = templates.replace(template, iter_=gen.iter, target=gen.target, body=body)
        return initialization + body

def transform(node, ctx):
    if False:
        while True:
            i = 10
    return ListCompTransformer(ctx).visit(node)