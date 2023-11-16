"""Lowers break statements to conditionals."""
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import templates
from nvidia.dali._autograph.pyct.static_analysis import activity
from nvidia.dali._autograph.pyct.static_analysis.annos import NodeAnno

class _Break(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.used = False
        self.control_var_name = None

    def __repr__(self):
        if False:
            return 10
        return 'used: %s, var: %s' % (self.used, self.control_var_name)

class BreakTransformer(converter.Base):
    """Canonicalizes break statements into additional conditionals."""

    def visit_Break(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.state[_Break].used = True
        var_name = self.state[_Break].control_var_name
        template = '\n      var_name = True\n      continue\n    '
        return templates.replace(template, var_name=var_name)

    def _guard_if_present(self, block, var_name):
        if False:
            print('Hello World!')
        'Prevents the block from executing if var_name is set.'
        if not block:
            return block
        template = '\n        if not var_name:\n          block\n      '
        node = templates.replace(template, var_name=var_name, block=block)
        return node

    def _process_body(self, nodes, break_var):
        if False:
            for i in range(10):
                print('nop')
        self.state[_Break].enter()
        self.state[_Break].control_var_name = break_var
        nodes = self.visit_block(nodes)
        break_used = self.state[_Break].used
        self.state[_Break].exit()
        return (nodes, break_used)

    def visit_While(self, node):
        if False:
            print('Hello World!')
        original_node = node
        scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
        break_var = self.ctx.namer.new_symbol('break_', scope.referenced)
        node.test = self.visit(node.test)
        (node.body, break_used) = self._process_body(node.body, break_var)
        node.orelse = self.visit_block(node.orelse)
        if not break_used:
            template = '\n        while test:\n          body\n        orelse\n      '
            node = templates.replace(template, test=node.test, body=node.body, orelse=node.orelse)
            new_while_node = node[0]
            anno.copyanno(original_node, new_while_node, anno.Basic.DIRECTIVES)
            return node
        guarded_orelse = self._guard_if_present(node.orelse, break_var)
        template = '\n      var_name = False\n      while not var_name and test:\n        body\n      orelse\n    '
        node = templates.replace(template, var_name=break_var, test=node.test, body=node.body, orelse=guarded_orelse)
        new_while_node = node[1]
        anno.copyanno(original_node, new_while_node, anno.Basic.DIRECTIVES)
        return node

    def visit_For(self, node):
        if False:
            for i in range(10):
                print('nop')
        original_node = node
        scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
        break_var = self.ctx.namer.new_symbol('break_', scope.referenced)
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        (node.body, break_used) = self._process_body(node.body, break_var)
        node.orelse = self.visit_block(node.orelse)
        if not break_used:
            template = '\n        for target in iter_:\n          body\n        orelse\n      '
            node = templates.replace(template, iter_=node.iter, target=node.target, body=node.body, orelse=node.orelse)
            new_for_node = node[0]
            anno.copyanno(original_node, new_for_node, anno.Basic.EXTRA_LOOP_TEST)
            anno.copyanno(original_node, new_for_node, anno.Basic.DIRECTIVES)
            return node
        guarded_orelse = self._guard_if_present(node.orelse, break_var)
        extra_test = templates.replace_as_expression('not var_name', var_name=break_var)
        template = '\n      var_name = False\n      for target in iter_:\n        (var_name,)\n        body\n      orelse\n    '
        node = templates.replace(template, var_name=break_var, iter_=node.iter, target=node.target, body=node.body, orelse=guarded_orelse)
        new_for_node = node[1]
        anno.setanno(new_for_node, anno.Basic.EXTRA_LOOP_TEST, extra_test)
        anno.copyanno(original_node, new_for_node, anno.Basic.DIRECTIVES)
        return node

def transform(node, ctx):
    if False:
        for i in range(10):
            print('nop')
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx, None)
    transformer = BreakTransformer(ctx)
    node = transformer.visit(node)
    return node