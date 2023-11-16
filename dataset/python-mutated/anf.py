"""Conversion to A-normal form.

The general idea of A-normal form is that every intermediate value is
explicitly named with a variable.  For more, see
https://en.wikipedia.org/wiki/A-normal_form.

The specific converters used here are based on Python AST semantics as
documented at https://greentreesnakes.readthedocs.io/en/latest/.
"""
import collections
import gast
import six
from nvidia.dali._autograph.pyct import gast_util
from nvidia.dali._autograph.pyct import templates
from nvidia.dali._autograph.pyct import transformer

class DummyGensym(object):
    """A dumb gensym that suffixes a stem by sequential numbers from 1000."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._idx = 0

    def new_name(self, stem='tmp'):
        if False:
            print('Hello World!')
        self._idx += 1
        return stem + '_' + str(1000 + self._idx)
REPLACE = lambda _1, _2, _3: True
LEAVE = lambda _1, _2, _3: False
ANY = object()

class ASTEdgePattern(collections.namedtuple('ASTEdgePattern', ['parent', 'field', 'child'])):
    """A pattern defining a type of AST edge.

  This consists of three components:
  - The type of the parent node, checked with isinstance,
  - The name of the field, checked with string equality, and
  - The type of the child node, also checked with isinstance.
  If all three match, the whole pattern is considered to match.

  In all three slots, the special value `anf.ANY` is treated as "match
  anything".  The internal nodes are produced from the `gast` library rather
  than the standard `ast` module, which may affect `isinstance` checks.
  """
    __slots__ = ()

    def matches(self, parent, field, child):
        if False:
            return 10
        'Computes whether this pattern matches the given edge.'
        if self.parent is ANY or isinstance(parent, self.parent):
            pass
        else:
            return False
        if self.field is ANY or field == self.field:
            pass
        else:
            return False
        return self.child is ANY or isinstance(child, self.child)

class AnfTransformer(transformer.Base):
    """Performs the conversion to A-normal form (ANF)."""

    def __init__(self, ctx, config):
        if False:
            for i in range(10):
                print('nop')
        'Creates an ANF transformer.\n\n    Args:\n      ctx: transformer.Context\n      config: Configuration\n    '
        super(AnfTransformer, self).__init__(ctx)
        if config is None:
            literal_node_types = (gast.Constant, gast.Name)
            self._overrides = [(ASTEdgePattern(ANY, ANY, literal_node_types), LEAVE), (ASTEdgePattern(ANY, ANY, gast.expr), REPLACE)]
        else:
            self._overrides = config
        self._gensym = DummyGensym()
        self._pending_statements = []

    def _consume_pending_statements(self):
        if False:
            return 10
        ans = self._pending_statements
        self._pending_statements = []
        return ans

    def _add_pending_statement(self, stmt):
        if False:
            while True:
                i = 10
        self._pending_statements.append(stmt)

    def _match(self, pattern, parent, field, child):
        if False:
            while True:
                i = 10
        if pattern is ANY:
            return True
        else:
            return pattern.matches(parent, field, child)

    def _should_transform(self, parent, field, child):
        if False:
            print('Hello World!')
        for (pat, result) in self._overrides:
            if self._match(pat, parent, field, child):
                return result(parent, field, child)
        return False

    def _do_transform_node(self, node):
        if False:
            while True:
                i = 10
        temp_name = self._gensym.new_name()
        temp_assign = templates.replace('temp_name = expr', temp_name=temp_name, expr=node)[0]
        self._add_pending_statement(temp_assign)
        answer = templates.replace('temp_name', temp_name=temp_name)[0]
        return answer

    def _ensure_node_in_anf(self, parent, field, node):
        if False:
            i = 10
            return i + 15
        'Puts `node` in A-normal form, by replacing it with a variable if needed.\n\n    The exact definition of A-normal form is given by the configuration.  The\n    parent and the incoming field name are only needed because the configuration\n    may be context-dependent.\n\n    Args:\n      parent: An AST node, the parent of `node`.\n      field: The field name under which `node` is the child of `parent`.\n      node: An AST node, potentially to be replaced with a variable reference.\n\n    Returns:\n      node: An AST node; the argument if transformation was not necessary,\n        or the new variable reference if it was.\n    '
        if node is None:
            return node
        if _is_trivial(node):
            return node
        if isinstance(node, list):
            return [self._ensure_node_in_anf(parent, field, n) for n in node]
        if isinstance(node, gast.keyword):
            node.value = self._ensure_node_in_anf(parent, field, node.value)
            return node
        if isinstance(node, (gast.Starred, gast.withitem, gast.slice)):
            return self._ensure_fields_in_anf(node, parent, field)
        if self._should_transform(parent, field, node):
            return self._do_transform_node(node)
        else:
            return node

    def _ensure_fields_in_anf(self, node, parent=None, super_field=None):
        if False:
            return 10
        for field in node._fields:
            if field.startswith('__'):
                continue
            parent_supplied = node if parent is None else parent
            field_supplied = field if super_field is None else super_field
            setattr(node, field, self._ensure_node_in_anf(parent_supplied, field_supplied, getattr(node, field)))
        return node

    def _visit_strict_statement(self, node, children_ok_to_transform=True):
        if False:
            for i in range(10):
                print('nop')
        assert not self._pending_statements
        node = self.generic_visit(node)
        if children_ok_to_transform:
            self._ensure_fields_in_anf(node)
        results = self._consume_pending_statements()
        results.append(node)
        return results

    def _visit_trivial_only_statement(self, node, msg):
        if False:
            return 10
        assert not self._pending_statements
        node = self.generic_visit(node)
        self._ensure_fields_in_anf(node)
        if self._pending_statements:
            raise ValueError(msg)
        else:
            return node

    def _visit_strict_expression(self, node):
        if False:
            print('Hello World!')
        node = self.generic_visit(node)
        self._ensure_fields_in_anf(node)
        return node

    def _visit_trivial_only_expression(self, node, msg):
        if False:
            return 10
        k = len(self._pending_statements)
        node = self.generic_visit(node)
        self._ensure_fields_in_anf(node)
        if len(self._pending_statements) != k:
            raise ValueError(msg)
        else:
            return node

    def visit_Return(self, node):
        if False:
            print('Hello World!')
        return self._visit_strict_statement(node)

    def visit_Delete(self, node):
        if False:
            return 10
        return self._visit_strict_statement(node, children_ok_to_transform=False)

    def visit_Assign(self, node):
        if False:
            i = 10
            return i + 15
        return self._visit_strict_statement(node, children_ok_to_transform=False)

    def visit_AugAssign(self, node):
        if False:
            print('Hello World!')
        return self._visit_strict_statement(node, children_ok_to_transform=False)

    def visit_Print(self, node):
        if False:
            i = 10
            return i + 15
        return self._visit_strict_statement(node)

    def visit_For(self, node):
        if False:
            while True:
                i = 10
        assert not self._pending_statements
        self.visit(node.iter)
        node.iter = self._ensure_node_in_anf(node, 'iter', node.iter)
        iter_stmts = self._consume_pending_statements()
        node = self.generic_visit(node)
        assert not self._pending_statements
        iter_stmts.append(node)
        return iter_stmts

    def visit_AsyncFor(self, node):
        if False:
            i = 10
            return i + 15
        msg = 'Nontrivial AsyncFor nodes not supported yet (need to think through the semantics).'
        return self._visit_trivial_only_statement(node, msg)

    def visit_While(self, node):
        if False:
            print('Hello World!')
        assert not self._pending_statements
        self.visit(node.test)
        node.test = self._ensure_node_in_anf(node, 'test', node.test)
        if self._pending_statements:
            msg = 'While with nontrivial test not supported yet (need to avoid precomputing the test).'
            raise ValueError(msg)
        return self.generic_visit(node)

    def visit_If(self, node):
        if False:
            return 10
        assert not self._pending_statements
        self.visit(node.test)
        node.test = self._ensure_node_in_anf(node, 'test', node.test)
        condition_stmts = self._consume_pending_statements()
        node = self.generic_visit(node)
        assert not self._pending_statements
        condition_stmts.append(node)
        return condition_stmts

    def visit_With(self, node):
        if False:
            while True:
                i = 10
        assert not self._pending_statements
        for item in node.items:
            self.visit(item)
        node.items = [self._ensure_node_in_anf(node, 'items', n) for n in node.items]
        contexts_stmts = self._consume_pending_statements()
        node = self.generic_visit(node)
        assert not self._pending_statements
        contexts_stmts.append(node)
        return contexts_stmts

    def visit_AsyncWith(self, node):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Nontrivial AsyncWith nodes not supported yet (need to think through the semantics).'
        return self._visit_trivial_only_statement(node, msg)

    def visit_Raise(self, node):
        if False:
            i = 10
            return i + 15
        return self._visit_strict_statement(node)

    def visit_Assert(self, node):
        if False:
            print('Hello World!')
        msg = 'Nontrivial Assert nodes not supported yet (need to avoid computing the test when assertions are off, and avoid computing the irritant when the assertion does not fire).'
        return self._visit_trivial_only_statement(node, msg)

    def visit_Exec(self, node):
        if False:
            return 10
        return self._visit_strict_statement(node)

    def visit_Expr(self, node):
        if False:
            return 10
        return self._visit_strict_statement(node, children_ok_to_transform=False)

    def visit_BoolOp(self, node):
        if False:
            return 10
        msg = 'Nontrivial BoolOp nodes not supported yet (need to preserve short-circuiting semantics).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_BinOp(self, node):
        if False:
            print('Hello World!')
        return self._visit_strict_expression(node)

    def visit_UnaryOp(self, node):
        if False:
            i = 10
            return i + 15
        return self._visit_strict_expression(node)

    def visit_Lambda(self, node):
        if False:
            i = 10
            return i + 15
        msg = 'Nontrivial Lambda nodes not supported (cannot insert statements into lambda bodies).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_IfExp(self, node):
        if False:
            i = 10
            return i + 15
        msg = 'Nontrivial IfExp nodes not supported yet (need to convert to If statement, to evaluate branches lazily and insert statements into them).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_Dict(self, node):
        if False:
            while True:
                i = 10
        return self._visit_strict_expression(node)

    def visit_Set(self, node):
        if False:
            print('Hello World!')
        return self._visit_strict_expression(node)

    def visit_ListComp(self, node):
        if False:
            for i in range(10):
                print('nop')
        msg = 'ListComp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
        raise ValueError(msg)

    def visit_SetComp(self, node):
        if False:
            for i in range(10):
                print('nop')
        msg = 'SetComp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
        raise ValueError(msg)

    def visit_DictComp(self, node):
        if False:
            return 10
        msg = 'DictComp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
        raise ValueError(msg)

    def visit_GeneratorExp(self, node):
        if False:
            while True:
                i = 10
        msg = 'GeneratorExp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
        raise ValueError(msg)

    def visit_Await(self, node):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Nontrivial Await nodes not supported yet (need to think through the semantics).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_Yield(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self._visit_strict_expression(node)

    def visit_YieldFrom(self, node):
        if False:
            print('Hello World!')
        msg = 'Nontrivial YieldFrom nodes not supported yet (need to unit-test them in Python 2).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_Compare(self, node):
        if False:
            return 10
        if len(node.ops) > 1:
            msg = 'Multi-ary compare nodes not supported yet (need to preserve short-circuiting semantics).'
            raise ValueError(msg)
        return self._visit_strict_expression(node)

    def visit_Call(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self._visit_strict_expression(node)

    def visit_Repr(self, node):
        if False:
            print('Hello World!')
        msg = 'Nontrivial Repr nodes not supported yet (need to research their syntax and semantics).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_FormattedValue(self, node):
        if False:
            return 10
        msg = 'Nontrivial FormattedValue nodes not supported yet (need to unit-test them in Python 2).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_JoinedStr(self, node):
        if False:
            print('Hello World!')
        msg = 'Nontrivial JoinedStr nodes not supported yet (need to unit-test them in Python 2).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_Attribute(self, node):
        if False:
            i = 10
            return i + 15
        return self._visit_strict_expression(node)

    def visit_Subscript(self, node):
        if False:
            i = 10
            return i + 15
        return self._visit_strict_expression(node)

    def visit_List(self, node):
        if False:
            print('Hello World!')
        node = self.generic_visit(node)
        if not isinstance(node.ctx, gast.Store):
            self._ensure_fields_in_anf(node)
        return node

    def visit_Tuple(self, node):
        if False:
            return 10
        node = self.generic_visit(node)
        if not isinstance(node.ctx, gast.Store):
            self._ensure_fields_in_anf(node)
        return node

def _is_py2_name_constant(node):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(node, gast.Name) and node.id in ['True', 'False', 'None']

def _is_trivial(node):
    if False:
        print('Hello World!')
    "Returns whether to consider the given node 'trivial'.\n\n  The definition of 'trivial' is a node that can't meaningfully be pulled out\n  into its own assignment statement.\n\n  This is surprisingly difficult to do robustly across versions of Python and\n  gast, as the parsing of constants has changed, if I may, constantly.\n\n  Args:\n    node: An AST node to check for triviality\n\n  Returns:\n    trivial: A Python `bool` indicating whether the node is trivial.\n  "
    trivial_node_types = (gast.Name, bool, six.string_types, gast.Add, gast.Sub, gast.Mult, gast.Div, gast.Mod, gast.Pow, gast.LShift, gast.RShift, gast.BitOr, gast.BitXor, gast.BitAnd, gast.FloorDiv, gast.Invert, gast.Not, gast.UAdd, gast.USub, gast.Eq, gast.NotEq, gast.Lt, gast.LtE, gast.Gt, gast.GtE, gast.Is, gast.IsNot, gast.In, gast.NotIn, gast.expr_context)
    if isinstance(node, trivial_node_types) and (not _is_py2_name_constant(node)):
        return True
    if gast_util.is_ellipsis(node):
        return True
    return False

def transform(node, ctx, config=None):
    if False:
        for i in range(10):
            print('nop')
    'Converts the given node to A-normal form (ANF).\n\n  The general idea of A-normal form: https://en.wikipedia.org/wiki/A-normal_form\n\n  The specific converters used here are based on Python AST semantics as\n  documented at https://greentreesnakes.readthedocs.io/en/latest/.\n\n  What exactly should be considered A-normal form for any given programming\n  language is not completely obvious.  The transformation defined here is\n  therefore configurable as to which syntax to replace with a fresh variable and\n  which to leave be.  The configuration is intentionally flexible enough to\n  define very precise variable insertion transformations, should that be\n  desired.\n\n  The configuration is a list of syntax rules, each of which is a 2-tuple:\n  - An `ASTEdgePattern` (which see) defining a type of AST edge, and\n  - Whether to transform children of such edges.\n  The special object `anf.ANY` may be used as a pattern that matches all edges.\n\n  Each replacement directive is one of three possible things:\n  - The object `anf.REPLACE`, meaning "Replace this child node with a variable",\n  - The object `anf.LEAVE`, meaning "Do not replace this child node with a\n    variable", or\n  - A Python callable.  If a callable, it is called with the parent node, the\n    field name, and the child node, and must compute a boolean indicating\n    whether to transform the child node or not.  The callable is free to use\n    whatever context information it chooses.  The callable may be invoked more\n    than once on the same link, and must produce the same answer each time.\n\n  The syntax rules are tested in order, and the first match governs.  If no rule\n  matches, the node is not transformed.\n\n  The above rules notwithstanding,\n  - Variable references are never replaced with (fresh) variables, as that would\n    accomplish nothing.\n  - The left-hand children of Assign and AugAssign nodes, and the children of\n    Del nodes, are never replaced with variables, as that would break their\n    semantics.\n  - The right-hand children of Assign nodes are never replaced with variables,\n    as the original assignment would still have to be present in the result\n    to define the new variable.  (That is, there\'s no point in transforming\n    `x = sin(y)` into `tmp = sin(y); x = tmp`.)\n  - The right-hand children of AugAssign nodes are never replaced with variables\n    either, but only because the difference from Assign was considered a\n    potential source of confusion (and it would have been slightly awkward in\n    the code to treat the RHS differently than the LHS).\n  - Various special-purpose AST nodes are not exposed to the configuration, lest\n    the transform produce invalid syntax like, e.g., `tmp = +; x = 1 tmp 2`.\n\n  For example, the configuration\n  ```python\n  [(anf.ASTEdgePattern(anf.ANY, anf.ANY, gast.expr), anf.REPLACE)]\n  ```\n  gives explicit fresh names to all expressions regardless of context (except as\n  outlined above), whereas\n  ```python\n  [(anf.ASTEdgePattern(gast.If, "test", anf.ANY), anf.REPLACE)]\n  ```\n  only transforms the conditionals of `if` statements (but not, e.g., `while`).\n\n  If no configuration is supplied, the default behavior is to transform all\n  expressions except literal constants, which is defined as a configuration as\n  ```python\n  # For Python 3, and gast library versions before 0.3\n  literals = (gast.Num, gast.Str, gast.Bytes, gast.NameConstant)\n  [(anf.ASTEdgePattern(anf.ANY, anf.ANY, literals), anf.LEAVE),\n   (anf.ASTEdgePattern(anf.ANY, anf.ANY, gast.expr), anf.REPLACE)]\n  ```\n\n  Args:\n    node: The node to transform.\n    ctx: transformer.EntityInfo.  TODO(mdan): What information does this\n      argument provide?\n    config: Optional ANF configuration.  If omitted, ANF replaces all expression\n      expect literal constants.\n  '
    return AnfTransformer(ctx, config).visit(node)