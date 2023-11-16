"""Handles function calls, by generating compiled function names and calls.

Note: this transformer does not rename the top level object being converted;
that is the caller's responsibility.

Requires function_scopes.
"""
import gast
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import templates
from nvidia.dali._autograph.utils import ag_logging

class _Function(object):
    no_root = True

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.context_name = None
set_trace_warned = False

class _ArgTemplateBuilder(object):
    """Constructs a tuple representing the positional arguments in a call.

  Example (yes, it's legal Python 3):

      f(*args1, b, *args2, c, d)  ->  args1 + (b,) + args2 + (c, d)
  """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._arg_accumulator = []
        self._argspec = []
        self._finalized = False

    def _consume_args(self):
        if False:
            i = 10
            return i + 15
        if self._arg_accumulator:
            self._argspec.append(gast.Tuple(elts=self._arg_accumulator, ctx=gast.Load()))
            self._arg_accumulator = []

    def add_arg(self, a):
        if False:
            i = 10
            return i + 15
        self._arg_accumulator.append(a)

    def add_stararg(self, a):
        if False:
            print('Hello World!')
        self._consume_args()
        self._argspec.append(gast.Call(gast.Name('tuple', ctx=gast.Load(), annotation=None, type_comment=None), args=[a], keywords=()))

    def finalize(self):
        if False:
            print('Hello World!')
        self._consume_args()
        self._finalized = True

    def to_ast(self):
        if False:
            for i in range(10):
                print('nop')
        assert self._finalized
        if self._argspec:
            result = self._argspec[0]
            for i in range(1, len(self._argspec)):
                result = gast.BinOp(result, gast.Add(), self._argspec[i])
            return result
        return gast.Tuple([], gast.Load())

class CallTreeTransformer(converter.Base):
    """Transforms the call tree by renaming transformed symbols."""

    def visit_Lambda(self, node):
        if False:
            return 10
        if not anno.hasanno(node, 'function_context_name'):
            return self.generic_visit(node)
        with self.state[_Function] as fn_scope:
            fn_scope.context_name = anno.getanno(node, 'function_context_name')
            return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.decorator_list = self.visit_block(node.decorator_list)
        node.args.defaults = self.visit_block(node.args.defaults)
        for (i, d) in enumerate(node.args.kw_defaults):
            if d is not None:
                node.args.kw_defaults[i] = self.visit(d)
        with self.state[_Function] as fn_scope:
            assert anno.hasanno(node, 'function_context_name'), 'The function_scopes converter always creates a scope for functions.'
            fn_scope.context_name = anno.getanno(node, 'function_context_name')
            node.body = self.visit_block(node.body)
            if node.returns:
                node.returns = self.visit(node.returns)
            return node

    def visit_With(self, node):
        if False:
            print('Hello World!')
        node.body = self.visit_block(node.body)
        return node

    def _args_to_tuple(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Ties together all positional and *arg arguments in a single tuple.'
        builder = _ArgTemplateBuilder()
        for a in node.args:
            if isinstance(a, gast.Starred):
                builder.add_stararg(a.value)
            else:
                builder.add_arg(a)
        builder.finalize()
        return builder.to_ast()

    def _kwargs_to_dict(self, node):
        if False:
            print('Hello World!')
        'Ties together all keyword and **kwarg arguments in a single dict.'
        if node.keywords:
            return gast.Call(gast.Name('dict', ctx=gast.Load(), annotation=None, type_comment=None), args=(), keywords=node.keywords)
        else:
            return parser.parse_expression('None')

    def visit_Call(self, node):
        if False:
            while True:
                i = 10
        full_name = str(anno.getanno(node.func, anno.Basic.QN, default=''))
        function_context_name = self.state[_Function].context_name
        node = self.generic_visit(node)
        if full_name.startswith('ag__.'):
            return node
        if full_name.startswith(function_context_name + '.'):
            return node
        if full_name in ('pdb.set_trace', 'ipdb.set_trace', 'breakpoint'):
            global set_trace_warned
            if not set_trace_warned:
                ag_logging.warning('Detected `pdb.set_trace()` in user code. The code generated by AutoGraph is not optimized for step-by-step debugging.')
                set_trace_warned = True
            return node
        if full_name == 'print' and (not self.ctx.user.options.uses(converter.Feature.BUILTIN_FUNCTIONS)):
            return node
        template = '\n      ag__.converted_call(func, args, kwargs, function_ctx)\n    '
        new_call = templates.replace_as_expression(template, func=node.func, args=self._args_to_tuple(node), kwargs=self._kwargs_to_dict(node), function_ctx=function_context_name)
        return new_call

def transform(node, ctx):
    if False:
        i = 10
        return i + 15
    'Transform function call to the compiled counterparts.\n\n  Args:\n    node: AST\n    ctx: EntityContext\n  Returns:\n    A tuple (node, new_names):\n        node: The transformed AST\n        new_names: set(string), containing any newly-generated names\n  '
    node = qual_names.resolve(node)
    node = CallTreeTransformer(ctx).visit(node)
    return node