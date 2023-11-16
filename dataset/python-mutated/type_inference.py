"""Type inference.

This analysis annotates all symbols nodes of an AST with type information
extracted from static sources:
 * type annotations
 * global and local symbols visible to the function at analysis time
 * literals

Important: This analysis is static, and does not detect dynamic type changes.
The analysis attempts to use the values of external symbols, if available. These
values are also considered static for the purpose of analysis.

Requires reaching function definitions analysis.
"""
import itertools
from typing import Any, Callable, Dict, Set
import gast
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import cfg
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import transformer
from nvidia.dali._autograph.pyct.static_analysis import activity
from nvidia.dali._autograph.pyct.static_analysis import annos

class Resolver(object):
    """Resolver objects handle the process of looking up actual names and types.

  Unless noted otherwise, all resolve_* methods:
    * have a first namespace argument, mapping string to actual values
    * have a second types_namespace argument, mapping string to actual inferred
      types
    * specify names as QN objects
    * specify types as a Set of inferred types

  Unless noted otherwise, all resolve_* methods must return either:
    * a set of `type` objects
    * None
  """

    def res_name(self, ns, types_ns, name):
        if False:
            for i in range(10):
                print('nop')
        'Resolves the type/value an external (e.g. closure, global) variable.\n\n    Args:\n      ns: namespace\n      types_ns: types namespace\n      name: symbol name\n    Returns:\n      Tuple (type, static_value). The first element is the type to use for\n      inferrence. The second is the static value to use. Return None to treat it\n      as unknown.\n    '
        raise NotImplementedError('subclasses must implement')

    def res_value(self, ns, value):
        if False:
            print('Hello World!')
        'Resolves the type a literal or static value.'
        raise NotImplementedError('subclasses must implement')

    def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        if False:
            i = 10
            return i + 15
        'Resolves the type of a (possibly annotated) function argument.\n\n    Args:\n      ns: namespace\n      types_ns: types namespace\n      f_name: str, the function name\n      name: str, the argument name\n      type_anno: the type annotating the argument, if any\n      f_is_local: bool, whether the function is a local function\n    Returns:\n      Set of the argument types.\n    '
        raise NotImplementedError('subclasses must implement')

    def res_call(self, ns, types_ns, node, f_type, args, keywords):
        if False:
            print('Hello World!')
        'Resolves the return type an external function or method call.\n\n    Args:\n      ns: namespace\n      types_ns: types namespace\n      node: str, the function name\n      f_type: types of the actual function being called, if known\n      args: types of each respective argument in node.args\n      keywords: types of each respective argument in node.keywords\n\n    Returns:\n      Tuple (return_type, side_effect_types). The first element is just the\n      return types of the function. The second element is a map from\n      argument names to sets of types, and allow modelling side effects of\n      functions (for example via global or nonlocal).\n    '
        raise NotImplementedError('subclasses must implement')

    def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        if False:
            i = 10
            return i + 15
        'Resolves the return type of slice operation.'
        raise NotImplementedError('subclasses must implement')

    def res_compare(self, ns, types_ns, node, left, right):
        if False:
            print('Hello World!')
        'Resolves the return type of a unary operation.'
        raise NotImplementedError('subclasses must implement')

    def res_unop(self, ns, types_ns, node, opnd):
        if False:
            print('Hello World!')
        'Resolves the return type of a unary operation.'
        raise NotImplementedError('subclasses must implement')

    def res_binop(self, ns, types_ns, node, left, right):
        if False:
            print('Hello World!')
        'Resolves the return type of a binary operation.'
        raise NotImplementedError('subclasses must implement')

    def res_list_literal(self, ns, elt_types):
        if False:
            while True:
                i = 10
        'Resolves the type of a list literal from its elements.'
        raise NotImplementedError('subclasses must implement')

class _TypeMap(object):
    """Abstraction for the state of the CFG walk for type inference.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    types: Dict[qual_names.QN, Set[Type]], mapping symbols to the set of
      possible types.
  """

    def __init__(self, init_from=None):
        if False:
            while True:
                i = 10
        if init_from:
            assert isinstance(init_from, _TypeMap)
            self.types = {s: set(other_types) for (s, other_types) in init_from.types.items()}
        else:
            self.types = {}

    def __eq__(self, other):
        if False:
            return 10
        if frozenset(self.types.keys()) != frozenset(other.types.keys()):
            return False
        ret = all((self.types[s] == other.types[s] for s in self.types))
        return ret

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def __or__(self, other):
        if False:
            return 10
        assert isinstance(other, _TypeMap)
        result = _TypeMap(self)
        for (s, other_types) in other.types.items():
            if s not in result.types:
                self_types = set()
                result.types[s] = self_types
            else:
                self_types = result.types[s]
            self_types.update(other_types)
        return result

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'SymbolTable {}'.format(self.types)
NO_VALUE = object()

class StmtInferrer(gast.NodeVisitor):
    """Runs type inference on a single AST statement.

  This visitor annotates most nodes with type information. It also sets types
  for the symbols modified by this statement in its types_out property.

  Note: this inferrer is able to capture side effects of functions, however,
  these side effects will not be applied to the current expression. Doing so
  would create too much of a dependence on the runtime's internal rules about
  execution order.
  Example:

    def f():
      nonlocal a
      a = 1
      return a

    a = 0.0
    b = f() + a  # a = float; side effect of f() ignored
    print(a)  # a = int; side effect of f() accounted for
  """

    def __init__(self, resolver: Resolver, scope: activity.Scope, namespace: Dict[qual_names.QN, Any], closure_types: Dict[qual_names.QN, Set[Any]], types_in: _TypeMap):
        if False:
            i = 10
            return i + 15
        self.resolver = resolver
        self.scope = scope
        self.namespace = namespace
        self.closure_types = closure_types
        self.types_in = types_in
        self.new_symbols = {}
        self.rtype = None

    def visit(self, node):
        if False:
            while True:
                i = 10
        types = super().visit(node)
        if __debug__:
            self._check_set(types)
        if types is not None:
            anno.setanno(node, anno.Static.TYPES, tuple(types))
        return types

    def _check_set(self, value):
        if False:
            print('Hello World!')
        if value is not None and (not isinstance(value, set)):
            raise ValueError('{} method expected to return set, got {}'.format(self.resolver, value))

    def visit_Constant(self, node):
        if False:
            i = 10
            return i + 15
        types = self.resolver.res_value(self.namespace, node.value)
        if __debug__:
            self._check_set(types)
        return types

    def _apply_unpacking(self, node):
        if False:
            while True:
                i = 10
        assert isinstance(node.ctx, gast.Store)
        if self.rtype is not None:
            original_stype = self.rtype
            i_type = self.resolver.res_value(self.namespace, 0)
            for (i, elt) in enumerate(node.elts):
                self.rtype = self.resolver.res_slice(self.namespace, self.types_in.types, i, original_stype, i_type)
                self.visit(elt)
            self.rtype = original_stype
            return original_stype
        return None

    def visit_Tuple(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.ctx, gast.Load):
            elt_types = ()
            for elt in node.elts:
                types_ = self.visit(elt)
                if types_ is None:
                    return None
                elt_types += (types_,)
            return set(itertools.product(*elt_types))
        return self._apply_unpacking(node)

    def visit_List(self, node):
        if False:
            return 10
        if isinstance(node.ctx, gast.Load):
            elt_types = tuple((self.visit(elt) for elt in node.elts))
            return self.resolver.res_list_literal(self.namespace, elt_types)
        return self._apply_unpacking(node)

    def visit_Set(self, node):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def visit_Name(self, node):
        if False:
            return 10
        name = anno.getanno(node, anno.Basic.QN)
        if isinstance(node.ctx, gast.Load):
            types = self.types_in.types.get(name, None)
            if types is None:
                if name not in self.scope.bound or name in self.scope.nonlocals:
                    if name in self.closure_types:
                        types = self.closure_types[name]
                    else:
                        (types, value) = self.resolver.res_name(self.namespace, self.types_in.types, name)
                        if value is not None:
                            anno.setanno(node, anno.Static.VALUE, value)
        elif isinstance(node.ctx, gast.Param):
            f_is_local = self.scope.parent.parent is not None
            type_name = anno.getanno(node.annotation, anno.Basic.QN, None)
            types = self.resolver.res_arg(self.namespace, self.types_in.types, self.scope.function_name, name, type_name, f_is_local)
            if types is not None:
                self.new_symbols[name] = types
        elif isinstance(node.ctx, gast.Store):
            if self.rtype is not None:
                self.new_symbols[name] = self.rtype
            types = self.rtype
        else:
            assert False, 'unknown ctx'
        if __debug__:
            self._check_set(types)
        return types

    def visit_Attribute(self, node):
        if False:
            i = 10
            return i + 15
        parent_types = self.visit(node.value)
        parent_value = anno.Static.VALUE.of(node.value, None)
        if parent_value is not None:
            static_value = getattr(parent_value, node.attr, NO_VALUE)
            if static_value is NO_VALUE:
                (types, static_value) = self.resolver.res_name(self.namespace, self.types_in, anno.Basic.QN.of(node))
                anno.setanno(node, anno.Static.VALUE, static_value)
                if __debug__:
                    self._check_set(types)
                return types
        else:
            if parent_types is None:
                return None
            inferred_values = [getattr(t, node.attr, None) for t in parent_types]
            if not inferred_values:
                return None
            static_value = inferred_values[0]
            if static_value is None:
                return None
            if any((v is not static_value for v in inferred_values[1:])):
                return None
        types = self.resolver.res_value(self.namespace, static_value)
        anno.setanno(node, anno.Static.VALUE, static_value)
        if __debug__:
            self._check_set(types)
        return types

    def visit_FunctionDef(self, node):
        if False:
            while True:
                i = 10
        f_name = qual_names.QN(node.name)
        if node.decorator_list:
            raise NotImplementedError('decorators: {}'.format(node.decorator_list))
        ret_types = None
        if node.returns:
            (ret_types, _) = self.resolver.res_name(self.namespace, self.types_in.types, anno.Basic.QN.of(node.returns))
            if __debug__:
                self._check_set(ret_types)
        if ret_types is None:
            ret_types = {Any}
        f_types = set()
        for rt in ret_types:
            f_types.add(Callable[[Any], rt])
        self.new_symbols[f_name] = f_types
        return None

    def _resolve_typed_callable(self, f_types, arg_types, keyword_types):
        if False:
            while True:
                i = 10
        ret_types = set()
        for t in f_types:
            if isinstance(t, Callable):
                args = t.__args__
                if args:
                    ret_types.add(args[-1])
                else:
                    ret_types.add(Any)
            else:
                raise NotImplementedError('callable type {}'.format(type(t)))
        side_effects = None
        return (ret_types, side_effects)

    def visit_Call(self, node):
        if False:
            i = 10
            return i + 15
        self.visit(node.func)
        f_name = anno.Basic.QN.of(node.func)
        arg_types = [self.visit(a) for a in node.args]
        keyword_types = [self.visit(kw.value) for kw in node.keywords]
        if f_name in self.scope.bound:
            f_type = self.types_in.types.get(f_name, None)
            if f_type is None:
                (ret_type, side_effects) = (None, None)
            else:
                (ret_type, side_effects) = self._resolve_typed_callable(f_type, arg_types, keyword_types)
        else:
            f_type = anno.Static.TYPES.of(node.func, None)
            (ret_type, side_effects) = self.resolver.res_call(self.namespace, self.types_in.types, node, f_type, arg_types, keyword_types)
        if __debug__:
            self._check_set(ret_type)
            if side_effects:
                if not isinstance(side_effects, dict):
                    raise ValueError('side effects must be dict, got {}'.format(side_effects))
                for (k, v) in side_effects.items():
                    if not isinstance(k, qual_names.QN):
                        raise ValueError('side effect keys must be QNs, got {}'.format(k))
                    self._check_set(v)
        if side_effects:
            self.new_symbols.update(side_effects)
        return ret_type

    def visit_Expr(self, node):
        if False:
            return 10
        return self.visit(node.value)

    def visit_Assign(self, node):
        if False:
            while True:
                i = 10
        self.rtype = self.visit(node.value)
        for t in node.targets:
            self.visit(t)
        self.rtype = None

    def visit_Subscript(self, node):
        if False:
            while True:
                i = 10
        val_types = self.visit(node.value)
        slice_types = self.visit(node.slice)
        if val_types is None or slice_types is None:
            return None
        types = self.resolver.res_slice(self.namespace, self.types_in.types, node, val_types, slice_types)
        if __debug__:
            self._check_set(types)
        return types

    def visit_Compare(self, node):
        if False:
            return 10
        left_types = self.visit(node.left)
        right_types = [self.visit(c) for c in node.comparators]
        if left_types is None or any((t is None for t in right_types)):
            return None
        types = self.resolver.res_compare(self.namespace, self.types_in.types, node, left_types, right_types)
        if __debug__:
            self._check_set(types)
        return types

    def visit_BinOp(self, node):
        if False:
            while True:
                i = 10
        left_types = self.visit(node.left)
        right_types = self.visit(node.right)
        if left_types is None or right_types is None:
            return None
        types = self.resolver.res_binop(self.namespace, self.types_in.types, node, left_types, right_types)
        if __debug__:
            self._check_set(types)
        return types

    def visit_UnaryOp(self, node):
        if False:
            while True:
                i = 10
        opnd_types = self.visit(node.operand)
        if opnd_types is None:
            return None
        types = self.resolver.res_unop(self.namespace, self.types_in.types, node, opnd_types)
        if __debug__:
            self._check_set(types)
        return types

class Analyzer(cfg.GraphVisitor):
    """CFG visitor that propagates type information across statements."""

    def __init__(self, graph, resolver, namespace, scope, closure_types):
        if False:
            while True:
                i = 10
        'Creates a new analyzer.\n\n    Args:\n      graph: cfg.Graph\n      resolver: Resolver\n      namespace: Dict[str, Any]\n      scope: activity.Scope\n      closure_types: Dict[QN, Set]\n    '
        super(Analyzer, self).__init__(graph)
        self.resolver = resolver
        self.namespace = namespace
        self.scope = scope
        self.closure_types = closure_types
        context_types = {n: t for (n, t) in closure_types.items() if n not in scope.bound}
        if context_types:
            self.context_types = _TypeMap()
            self.context_types.types = context_types
        else:
            self.context_types = None

    def init_state(self, _):
        if False:
            for i in range(10):
                print('nop')
        return _TypeMap()

    def _update_closure_types(self, ast_node, types):
        if False:
            i = 10
            return i + 15
        existing_types = anno.Static.CLOSURE_TYPES.of(ast_node, None)
        if existing_types is None:
            existing_types = {}
            anno.Static.CLOSURE_TYPES.add_to(ast_node, existing_types)
        for (k, v) in types.types.items():
            if k in existing_types:
                existing_types[k].update(v)
            else:
                existing_types[k] = set(v)

    def visit_node(self, node):
        if False:
            for i in range(10):
                print('nop')
        prev_types_out = self.out[node]
        types_in = _TypeMap()
        for n in node.prev:
            types_in |= self.out[n]
        if self.context_types is not None and node is self.graph.entry:
            types_in |= self.context_types
        types_out = _TypeMap(types_in)
        ast_node = node.ast_node
        inferrer = StmtInferrer(self.resolver, self.scope, self.namespace, self.closure_types, types_in)
        inferrer.visit(ast_node)
        types_out.types.update(inferrer.new_symbols)
        reaching_fndefs = anno.Static.DEFINED_FNS_IN.of(ast_node)
        node_scope = anno.Static.SCOPE.of(ast_node, None)
        if node_scope is not None:
            reads = {str(qn) for qn in node_scope.read}
            for def_node in reaching_fndefs:
                if def_node.name in reads:
                    self._update_closure_types(def_node, types_out)
        self.in_[node] = types_in
        self.out[node] = types_out
        return prev_types_out != types_out

class FunctionVisitor(transformer.Base):
    """AST visitor that applies type inference to each function separately."""

    def __init__(self, source_info, graphs, resolver):
        if False:
            print('Hello World!')
        super(FunctionVisitor, self).__init__(source_info)
        self.graphs = graphs
        self.resolver = resolver

    def visit_FunctionDef(self, node):
        if False:
            return 10
        subgraph = self.graphs[node]
        scope = anno.getanno(node, annos.NodeAnno.ARGS_AND_BODY_SCOPE)
        closure_types = anno.getanno(node, anno.Static.CLOSURE_TYPES, {})
        analyzer = Analyzer(subgraph, self.resolver, self.ctx.info.namespace, scope, closure_types)
        analyzer.visit_forward()
        node.body = self.visit_block(node.body)
        return node

def resolve(node, source_info, graphs, resolver):
    if False:
        for i in range(10):
            print('nop')
    'Performs type inference.\n\n  Args:\n    node: ast.AST\n    source_info: transformer.SourceInfo\n    graphs: Dict[ast.FunctionDef, cfg.Graph]\n    resolver: Resolver\n\n  Returns:\n    ast.AST\n  '
    visitor = FunctionVisitor(source_info, graphs, resolver)
    node = visitor.visit(node)
    return node