"""Canonicalizes functions with multiple returns to use just one."""
import gast
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import templates
from nvidia.dali._autograph.pyct.static_analysis import activity
from nvidia.dali._autograph.pyct.static_analysis.annos import NodeAnno
BODY_DEFINITELY_RETURNS = 'BODY_DEFINITELY_RETURNS'
ORELSE_DEFINITELY_RETURNS = 'ORELSE_DEFINITELY_RETURNS'
STMT_DEFINITELY_RETURNS = 'STMT_DEFINITELY_RETURNS'

class _RewriteBlock(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.definitely_returns = False

class ConditionalReturnRewriter(converter.Base):
    """Rewrites a pattern where it's unobvious that all paths return a value.

  This rewrite allows avoiding intermediate None return values.

  The following pattern:

      if cond:
        <block 1>
        return
      else:
        <block 2>
      <block 3>

  is converted to:

      if cond:
        <block 1>
        return
      else:
        <block 2>
        <block 3>

  and vice-versa (if the else returns, subsequent statements are moved under the
  if branch).
  """

    def visit_Return(self, node):
        if False:
            print('Hello World!')
        self.state[_RewriteBlock].definitely_returns = True
        return node

    def _postprocess_statement(self, node):
        if False:
            i = 10
            return i + 15
        if anno.getanno(node, STMT_DEFINITELY_RETURNS, default=False):
            self.state[_RewriteBlock].definitely_returns = True
        if isinstance(node, gast.If) and anno.getanno(node, BODY_DEFINITELY_RETURNS, default=False):
            return (node, node.orelse)
        elif isinstance(node, gast.If) and anno.getanno(node, ORELSE_DEFINITELY_RETURNS, default=False):
            return (node, node.body)
        return (node, None)

    def _visit_statement_block(self, node, nodes):
        if False:
            for i in range(10):
                print('nop')
        self.state[_RewriteBlock].enter()
        new_nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
        block_definitely_returns = self.state[_RewriteBlock].definitely_returns
        self.state[_RewriteBlock].exit()
        return (new_nodes, block_definitely_returns)

    def visit_While(self, node):
        if False:
            while True:
                i = 10
        node.test = self.visit(node.test)
        (node.body, _) = self._visit_statement_block(node, node.body)
        (node.orelse, _) = self._visit_statement_block(node, node.orelse)
        return node

    def visit_For(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        (node.body, _) = self._visit_statement_block(node, node.body)
        (node.orelse, _) = self._visit_statement_block(node, node.orelse)
        return node

    def visit_With(self, node):
        if False:
            print('Hello World!')
        node.items = self.visit_block(node.items)
        (node.body, definitely_returns) = self._visit_statement_block(node, node.body)
        if definitely_returns:
            anno.setanno(node, STMT_DEFINITELY_RETURNS, True)
        return node

    def visit_Try(self, node):
        if False:
            for i in range(10):
                print('nop')
        (node.body, _) = self._visit_statement_block(node, node.body)
        (node.orelse, _) = self._visit_statement_block(node, node.orelse)
        (node.finalbody, _) = self._visit_statement_block(node, node.finalbody)
        node.handlers = self.visit_block(node.handlers)
        return node

    def visit_ExceptHandler(self, node):
        if False:
            return 10
        (node.body, _) = self._visit_statement_block(node, node.body)
        return node

    def visit_If(self, node):
        if False:
            i = 10
            return i + 15
        node.test = self.visit(node.test)
        (node.body, body_definitely_returns) = self._visit_statement_block(node, node.body)
        if body_definitely_returns:
            anno.setanno(node, BODY_DEFINITELY_RETURNS, True)
        (node.orelse, orelse_definitely_returns) = self._visit_statement_block(node, node.orelse)
        if orelse_definitely_returns:
            anno.setanno(node, ORELSE_DEFINITELY_RETURNS, True)
        if body_definitely_returns and orelse_definitely_returns:
            self.state[_RewriteBlock].definitely_returns = True
        return node

    def visit_FunctionDef(self, node):
        if False:
            return 10
        node.args = self.visit(node.args)
        (node.body, _) = self._visit_statement_block(node, node.body)
        return node

class _Block(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.is_function = False
        self.return_used = False
        self.create_guard_next = False
        self.create_guard_now = False

    def __repr__(self):
        if False:
            return 10
        return 'used: {}'.format(self.return_used)

class _Function(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.do_return_var_name = None
        self.retval_var_name = None

    def __repr__(self):
        if False:
            return 10
        return 'return control: {}, return value: {}'.format(self.do_return_var_name, self.retval_var_name)

class ReturnStatementsTransformer(converter.Base):
    """Lowers return statements into variables and conditionals.

  Specifically, the following pattern:

      <block 1>
      return val
      <block 2>

  is converted to:

      do_return = False
      retval = None

      <block 1>

      do_return = True
      retval = val

      if not do_return:
        <block 2>

      return retval

  The conversion adjusts loops as well:

      <block 1>
      while cond:
        <block 2>
        return retval

  is converted to:

      <block 1>
      while not do_return and cond:
        <block 2>
        do_return = True
        retval = val
  """

    def __init__(self, ctx, allow_missing_return):
        if False:
            print('Hello World!')
        super(ReturnStatementsTransformer, self).__init__(ctx)
        self.allow_missing_return = allow_missing_return

    def visit_Return(self, node):
        if False:
            print('Hello World!')
        for block in reversed(self.state[_Block].stack):
            block.return_used = True
            block.create_guard_next = True
            if block.is_function:
                break
        retval = node.value if node.value else parser.parse_expression('None')
        template = '\n      try:\n        do_return_var_name = True\n        retval_var_name = retval\n      except:\n        do_return_var_name = False\n        raise\n    '
        node = templates.replace(template, do_return_var_name=self.state[_Function].do_return_var_name, retval_var_name=self.state[_Function].retval_var_name, retval=retval)
        return node

    def _postprocess_statement(self, node):
        if False:
            while True:
                i = 10
        if not self.state[_Block].return_used:
            return (node, None)
        state = self.state[_Block]
        if state.create_guard_now:
            template = '\n        if not do_return_var_name:\n          original_node\n      '
            (cond,) = templates.replace(template, do_return_var_name=self.state[_Function].do_return_var_name, original_node=node)
            (node, block) = (cond, cond.body)
        else:
            (node, block) = (node, None)
        state.create_guard_now = state.create_guard_next
        state.create_guard_next = False
        return (node, block)

    def _visit_statement_block(self, node, nodes):
        if False:
            i = 10
            return i + 15
        self.state[_Block].enter()
        nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
        self.state[_Block].exit()
        return nodes

    def visit_While(self, node):
        if False:
            i = 10
            return i + 15
        node.test = self.visit(node.test)
        node.body = self._visit_statement_block(node, node.body)
        if self.state[_Block].return_used:
            node.test = templates.replace_as_expression('not control_var and test', test=node.test, control_var=self.state[_Function].do_return_var_name)
        node.orelse = self._visit_statement_block(node, node.orelse)
        return node

    def visit_For(self, node):
        if False:
            print('Hello World!')
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        node.body = self._visit_statement_block(node, node.body)
        if self.state[_Block].return_used:
            extra_test = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST, default=None)
            if extra_test is not None:
                extra_test = templates.replace_as_expression('not control_var and extra_test', extra_test=extra_test, control_var=self.state[_Function].do_return_var_name)
            else:
                extra_test = templates.replace_as_expression('not control_var', control_var=self.state[_Function].do_return_var_name)
            anno.setanno(node, anno.Basic.EXTRA_LOOP_TEST, extra_test)
        node.orelse = self._visit_statement_block(node, node.orelse)
        return node

    def visit_With(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.items = self.visit_block(node.items)
        node.body = self._visit_statement_block(node, node.body)
        return node

    def visit_Try(self, node):
        if False:
            print('Hello World!')
        node.body = self._visit_statement_block(node, node.body)
        node.orelse = self._visit_statement_block(node, node.orelse)
        node.finalbody = self._visit_statement_block(node, node.finalbody)
        node.handlers = self.visit_block(node.handlers)
        return node

    def visit_ExceptHandler(self, node):
        if False:
            return 10
        node.body = self._visit_statement_block(node, node.body)
        return node

    def visit_If(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.test = self.visit(node.test)
        node.body = self._visit_statement_block(node, node.body)
        node.orelse = self._visit_statement_block(node, node.orelse)
        return node

    def visit_FunctionDef(self, node):
        if False:
            print('Hello World!')
        with self.state[_Function] as fn:
            with self.state[_Block] as block:
                block.is_function = True
                scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
                do_return_var_name = self.ctx.namer.new_symbol('do_return', scope.referenced)
                retval_var_name = self.ctx.namer.new_symbol('retval_', scope.referenced)
                fn.do_return_var_name = do_return_var_name
                fn.retval_var_name = retval_var_name
                node.body = self._visit_statement_block(node, node.body)
                if block.return_used:
                    if self.allow_missing_return:
                        wrapper_node = node.body[-1]
                        assert isinstance(wrapper_node, gast.With), 'This transformer requires the functions converter.'
                        template = '\n              do_return_var_name = False\n              retval_var_name = ag__.UndefinedReturnValue()\n              body\n              return function_context.ret(retval_var_name, do_return_var_name)\n            '
                        wrapper_node.body = templates.replace(template, body=wrapper_node.body, do_return_var_name=do_return_var_name, function_context=anno.getanno(node, 'function_context_name'), retval_var_name=retval_var_name)
                    else:
                        template = '\n              body\n              return retval_var_name\n            '
                        node.body = templates.replace(template, body=node.body, do_return_var_name=do_return_var_name, retval_var_name=retval_var_name)
        return node

def transform(node, ctx, default_to_null_return=True):
    if False:
        while True:
            i = 10
    'Ensure a function has only a single return, at the end.'
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx, None)
    node = ConditionalReturnRewriter(ctx).visit(node)
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx, None)
    transformer = ReturnStatementsTransformer(ctx, allow_missing_return=default_to_null_return)
    node = transformer.visit(node)
    return node