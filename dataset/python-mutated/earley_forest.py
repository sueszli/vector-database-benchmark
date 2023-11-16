""""This module implements an SPPF implementation

This is used as the primary output mechanism for the Earley parser
in order to store complex ambiguities.

Full reference and more details is here:
https://web.archive.org/web/20190616123959/http://www.bramvandersanden.com/post/2014/06/shared-packed-parse-forest/
"""
from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree

class ForestNode:
    pass

class SymbolNode(ForestNode):
    """
    A Symbol Node represents a symbol (or Intermediate LR0).

    Symbol nodes are keyed by the symbol (s). For intermediate nodes
    s will be an LR0, stored as a tuple of (rule, ptr). For completed symbol
    nodes, s will be a string representing the non-terminal origin (i.e.
    the left hand side of the rule).

    The children of a Symbol or Intermediate Node will always be Packed Nodes;
    with each Packed Node child representing a single derivation of a production.

    Hence a Symbol Node with a single child is unambiguous.

    Parameters:
        s: A Symbol, or a tuple of (rule, ptr) for an intermediate node.
        start: The index of the start of the substring matched by this symbol (inclusive).
        end: The index of the end of the substring matched by this symbol (exclusive).

    Properties:
        is_intermediate: True if this node is an intermediate node.
        priority: The priority of the node's symbol.
    """
    Set: Type[AbstractSet] = set
    __slots__ = ('s', 'start', 'end', '_children', 'paths', 'paths_loaded', 'priority', 'is_intermediate', '_hash')

    def __init__(self, s, start, end):
        if False:
            while True:
                i = 10
        self.s = s
        self.start = start
        self.end = end
        self._children = self.Set()
        self.paths = self.Set()
        self.paths_loaded = False
        self.priority = float('-inf')
        self.is_intermediate = isinstance(s, tuple)
        self._hash = hash((self.s, self.start, self.end))

    def add_family(self, lr0, rule, start, left, right):
        if False:
            i = 10
            return i + 15
        self._children.add(PackedNode(self, lr0, rule, start, left, right))

    def add_path(self, transitive, node):
        if False:
            for i in range(10):
                print('nop')
        self.paths.add((transitive, node))

    def load_paths(self):
        if False:
            print('Hello World!')
        for (transitive, node) in self.paths:
            if transitive.next_titem is not None:
                vn = type(self)(transitive.next_titem.s, transitive.next_titem.start, self.end)
                vn.add_path(transitive.next_titem, node)
                self.add_family(transitive.reduction.rule.origin, transitive.reduction.rule, transitive.reduction.start, transitive.reduction.node, vn)
            else:
                self.add_family(transitive.reduction.rule.origin, transitive.reduction.rule, transitive.reduction.start, transitive.reduction.node, node)
        self.paths_loaded = True

    @property
    def is_ambiguous(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if this node is ambiguous.'
        return len(self.children) > 1

    @property
    def children(self):
        if False:
            return 10
        "Returns a list of this node's children sorted from greatest to\n        least priority."
        if not self.paths_loaded:
            self.load_paths()
        return sorted(self._children, key=attrgetter('sort_key'))

    def __iter__(self):
        if False:
            return 10
        return iter(self._children)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, SymbolNode):
            return False
        return self is other or (type(self.s) == type(other.s) and self.s == other.s and (self.start == other.start) and (self.end is other.end))

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return self._hash

    def __repr__(self):
        if False:
            return 10
        if self.is_intermediate:
            rule = self.s[0]
            ptr = self.s[1]
            before = (expansion.name for expansion in rule.expansion[:ptr])
            after = (expansion.name for expansion in rule.expansion[ptr:])
            symbol = '{} ::= {}* {}'.format(rule.origin.name, ' '.join(before), ' '.join(after))
        else:
            symbol = self.s.name
        return '({}, {}, {}, {})'.format(symbol, self.start, self.end, self.priority)

class StableSymbolNode(SymbolNode):
    """A version of SymbolNode that uses OrderedSet for output stability"""
    Set = OrderedSet

class PackedNode(ForestNode):
    """
    A Packed Node represents a single derivation in a symbol node.

    Parameters:
        rule: The rule associated with this node.
        parent: The parent of this node.
        left: The left child of this node. ``None`` if one does not exist.
        right: The right child of this node. ``None`` if one does not exist.
        priority: The priority of this node.
    """
    __slots__ = ('parent', 's', 'rule', 'start', 'left', 'right', 'priority', '_hash')

    def __init__(self, parent, s, rule, start, left, right):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.s = s
        self.start = start
        self.rule = rule
        self.left = left
        self.right = right
        self.priority = float('-inf')
        self._hash = hash((self.left, self.right))

    @property
    def is_empty(self):
        if False:
            return 10
        return self.left is None and self.right is None

    @property
    def sort_key(self):
        if False:
            print('Hello World!')
        '\n        Used to sort PackedNode children of SymbolNodes.\n        A SymbolNode has multiple PackedNodes if it matched\n        ambiguously. Hence, we use the sort order to identify\n        the order in which ambiguous children should be considered.\n        '
        return (self.is_empty, -self.priority, self.rule.order)

    @property
    def children(self):
        if False:
            i = 10
            return i + 15
        "Returns a list of this node's children."
        return [x for x in [self.left, self.right] if x is not None]

    def __iter__(self):
        if False:
            print('Hello World!')
        yield self.left
        yield self.right

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, PackedNode):
            return False
        return self is other or (self.left == other.left and self.right == other.right)

    def __hash__(self):
        if False:
            return 10
        return self._hash

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.s, tuple):
            rule = self.s[0]
            ptr = self.s[1]
            before = (expansion.name for expansion in rule.expansion[:ptr])
            after = (expansion.name for expansion in rule.expansion[ptr:])
            symbol = '{} ::= {}* {}'.format(rule.origin.name, ' '.join(before), ' '.join(after))
        else:
            symbol = self.s.name
        return '({}, {}, {}, {})'.format(symbol, self.start, self.priority, self.rule.order)

class TokenNode(ForestNode):
    """
    A Token Node represents a matched terminal and is always a leaf node.

    Parameters:
        token: The Token associated with this node.
        term: The TerminalDef matched by the token.
        priority: The priority of this node.
    """
    __slots__ = ('token', 'term', 'priority', '_hash')

    def __init__(self, token, term, priority=None):
        if False:
            i = 10
            return i + 15
        self.token = token
        self.term = term
        if priority is not None:
            self.priority = priority
        else:
            self.priority = term.priority if term is not None else 0
        self._hash = hash(token)

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, TokenNode):
            return False
        return self is other or self.token == other.token

    def __hash__(self):
        if False:
            while True:
                i = 10
        return self._hash

    def __repr__(self):
        if False:
            while True:
                i = 10
        return repr(self.token)

class ForestVisitor:
    """
    An abstract base class for building forest visitors.

    This class performs a controllable depth-first walk of an SPPF.
    The visitor will not enter cycles and will backtrack if one is encountered.
    Subclasses are notified of cycles through the ``on_cycle`` method.

    Behavior for visit events is defined by overriding the
    ``visit*node*`` functions.

    The walk is controlled by the return values of the ``visit*node_in``
    methods. Returning a node(s) will schedule them to be visited. The visitor
    will begin to backtrack if no nodes are returned.

    Parameters:
        single_visit: If ``True``, non-Token nodes will only be visited once.
    """

    def __init__(self, single_visit=False):
        if False:
            print('Hello World!')
        self.single_visit = single_visit

    def visit_token_node(self, node):
        if False:
            return 10
        'Called when a ``Token`` is visited. ``Token`` nodes are always leaves.'
        pass

    def visit_symbol_node_in(self, node):
        if False:
            while True:
                i = 10
        'Called when a symbol node is visited. Nodes that are returned\n        will be scheduled to be visited. If ``visit_intermediate_node_in``\n        is not implemented, this function will be called for intermediate\n        nodes as well.'
        pass

    def visit_symbol_node_out(self, node):
        if False:
            return 10
        'Called after all nodes returned from a corresponding ``visit_symbol_node_in``\n        call have been visited. If ``visit_intermediate_node_out``\n        is not implemented, this function will be called for intermediate\n        nodes as well.'
        pass

    def visit_packed_node_in(self, node):
        if False:
            while True:
                i = 10
        'Called when a packed node is visited. Nodes that are returned\n        will be scheduled to be visited. '
        pass

    def visit_packed_node_out(self, node):
        if False:
            print('Hello World!')
        'Called after all nodes returned from a corresponding ``visit_packed_node_in``\n        call have been visited.'
        pass

    def on_cycle(self, node, path):
        if False:
            return 10
        'Called when a cycle is encountered.\n\n        Parameters:\n            node: The node that causes a cycle.\n            path: The list of nodes being visited: nodes that have been\n                entered but not exited. The first element is the root in a forest\n                visit, and the last element is the node visited most recently.\n                ``path`` should be treated as read-only.\n        '
        pass

    def get_cycle_in_path(self, node, path):
        if False:
            while True:
                i = 10
        'A utility function for use in ``on_cycle`` to obtain a slice of\n        ``path`` that only contains the nodes that make up the cycle.'
        index = len(path) - 1
        while id(path[index]) != id(node):
            index -= 1
        return path[index:]

    def visit(self, root):
        if False:
            i = 10
            return i + 15
        visiting = set()
        visited = set()
        path = []
        input_stack = deque([root])
        vpno = getattr(self, 'visit_packed_node_out')
        vpni = getattr(self, 'visit_packed_node_in')
        vsno = getattr(self, 'visit_symbol_node_out')
        vsni = getattr(self, 'visit_symbol_node_in')
        vino = getattr(self, 'visit_intermediate_node_out', vsno)
        vini = getattr(self, 'visit_intermediate_node_in', vsni)
        vtn = getattr(self, 'visit_token_node')
        oc = getattr(self, 'on_cycle')
        while input_stack:
            current = next(reversed(input_stack))
            try:
                next_node = next(current)
            except StopIteration:
                input_stack.pop()
                continue
            except TypeError:
                pass
            else:
                if next_node is None:
                    continue
                if id(next_node) in visiting:
                    oc(next_node, path)
                    continue
                input_stack.append(next_node)
                continue
            if isinstance(current, TokenNode):
                vtn(current.token)
                input_stack.pop()
                continue
            current_id = id(current)
            if current_id in visiting:
                if isinstance(current, PackedNode):
                    vpno(current)
                elif current.is_intermediate:
                    vino(current)
                else:
                    vsno(current)
                input_stack.pop()
                path.pop()
                visiting.remove(current_id)
                visited.add(current_id)
            elif self.single_visit and current_id in visited:
                input_stack.pop()
            else:
                visiting.add(current_id)
                path.append(current)
                if isinstance(current, PackedNode):
                    next_node = vpni(current)
                elif current.is_intermediate:
                    next_node = vini(current)
                else:
                    next_node = vsni(current)
                if next_node is None:
                    continue
                if not isinstance(next_node, ForestNode):
                    next_node = iter(next_node)
                elif id(next_node) in visiting:
                    oc(next_node, path)
                    continue
                input_stack.append(next_node)

class ForestTransformer(ForestVisitor):
    """The base class for a bottom-up forest transformation. Most users will
    want to use ``TreeForestTransformer`` instead as it has a friendlier
    interface and covers most use cases.

    Transformations are applied via inheritance and overriding of the
    ``transform*node`` methods.

    ``transform_token_node`` receives a ``Token`` as an argument.
    All other methods receive the node that is being transformed and
    a list of the results of the transformations of that node's children.
    The return value of these methods are the resulting transformations.

    If ``Discard`` is raised in a node's transformation, no data from that node
    will be passed to its parent's transformation.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(ForestTransformer, self).__init__()
        self.data = dict()
        self.node_stack = deque()

    def transform(self, root):
        if False:
            for i in range(10):
                print('nop')
        'Perform a transformation on an SPPF.'
        self.node_stack.append('result')
        self.data['result'] = []
        self.visit(root)
        assert len(self.data['result']) <= 1
        if self.data['result']:
            return self.data['result'][0]

    def transform_symbol_node(self, node, data):
        if False:
            while True:
                i = 10
        'Transform a symbol node.'
        return node

    def transform_intermediate_node(self, node, data):
        if False:
            return 10
        'Transform an intermediate node.'
        return node

    def transform_packed_node(self, node, data):
        if False:
            return 10
        'Transform a packed node.'
        return node

    def transform_token_node(self, node):
        if False:
            return 10
        'Transform a ``Token``.'
        return node

    def visit_symbol_node_in(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.node_stack.append(id(node))
        self.data[id(node)] = []
        return node.children

    def visit_packed_node_in(self, node):
        if False:
            return 10
        self.node_stack.append(id(node))
        self.data[id(node)] = []
        return node.children

    def visit_token_node(self, node):
        if False:
            for i in range(10):
                print('nop')
        transformed = self.transform_token_node(node)
        if transformed is not Discard:
            self.data[self.node_stack[-1]].append(transformed)

    def _visit_node_out_helper(self, node, method):
        if False:
            i = 10
            return i + 15
        self.node_stack.pop()
        transformed = method(node, self.data[id(node)])
        if transformed is not Discard:
            self.data[self.node_stack[-1]].append(transformed)
        del self.data[id(node)]

    def visit_symbol_node_out(self, node):
        if False:
            i = 10
            return i + 15
        self._visit_node_out_helper(node, self.transform_symbol_node)

    def visit_intermediate_node_out(self, node):
        if False:
            print('Hello World!')
        self._visit_node_out_helper(node, self.transform_intermediate_node)

    def visit_packed_node_out(self, node):
        if False:
            while True:
                i = 10
        self._visit_node_out_helper(node, self.transform_packed_node)

class ForestSumVisitor(ForestVisitor):
    """
    A visitor for prioritizing ambiguous parts of the Forest.

    This visitor is used when support for explicit priorities on
    rules is requested (whether normal, or invert). It walks the
    forest (or subsets thereof) and cascades properties upwards
    from the leaves.

    It would be ideal to do this during parsing, however this would
    require processing each Earley item multiple times. That's
    a big performance drawback; so running a forest walk is the
    lesser of two evils: there can be significantly more Earley
    items created during parsing than there are SPPF nodes in the
    final tree.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(ForestSumVisitor, self).__init__(single_visit=True)

    def visit_packed_node_in(self, node):
        if False:
            i = 10
            return i + 15
        yield node.left
        yield node.right

    def visit_symbol_node_in(self, node):
        if False:
            i = 10
            return i + 15
        return iter(node.children)

    def visit_packed_node_out(self, node):
        if False:
            for i in range(10):
                print('nop')
        priority = node.rule.options.priority if not node.parent.is_intermediate and node.rule.options.priority else 0
        priority += getattr(node.right, 'priority', 0)
        priority += getattr(node.left, 'priority', 0)
        node.priority = priority

    def visit_symbol_node_out(self, node):
        if False:
            while True:
                i = 10
        node.priority = max((child.priority for child in node.children))

class PackedData:
    """Used in transformationss of packed nodes to distinguish the data
    that comes from the left child and the right child.
    """

    class _NoData:
        pass
    NO_DATA = _NoData()

    def __init__(self, node, data):
        if False:
            for i in range(10):
                print('nop')
        self.left = self.NO_DATA
        self.right = self.NO_DATA
        if data:
            if node.left is not None:
                self.left = data[0]
                if len(data) > 1:
                    self.right = data[1]
            else:
                self.right = data[0]

class ForestToParseTree(ForestTransformer):
    """Used by the earley parser when ambiguity equals 'resolve' or
    'explicit'. Transforms an SPPF into an (ambiguous) parse tree.

    Parameters:
        tree_class: The tree class to use for construction
        callbacks: A dictionary of rules to functions that output a tree
        prioritizer: A ``ForestVisitor`` that manipulates the priorities of ForestNodes
        resolve_ambiguity: If True, ambiguities will be resolved based on
                        priorities. Otherwise, `_ambig` nodes will be in the resulting tree.
        use_cache: If True, the results of packed node transformations will be cached.
    """

    def __init__(self, tree_class=Tree, callbacks=dict(), prioritizer=ForestSumVisitor(), resolve_ambiguity=True, use_cache=True):
        if False:
            print('Hello World!')
        super(ForestToParseTree, self).__init__()
        self.tree_class = tree_class
        self.callbacks = callbacks
        self.prioritizer = prioritizer
        self.resolve_ambiguity = resolve_ambiguity
        self._use_cache = use_cache
        self._cache = {}
        self._on_cycle_retreat = False
        self._cycle_node = None
        self._successful_visits = set()

    def visit(self, root):
        if False:
            print('Hello World!')
        if self.prioritizer:
            self.prioritizer.visit(root)
        super(ForestToParseTree, self).visit(root)
        self._cache = {}

    def on_cycle(self, node, path):
        if False:
            while True:
                i = 10
        logger.debug('Cycle encountered in the SPPF at node: %s. As infinite ambiguities cannot be represented in a tree, this family of derivations will be discarded.', node)
        self._cycle_node = node
        self._on_cycle_retreat = True

    def _check_cycle(self, node):
        if False:
            print('Hello World!')
        if self._on_cycle_retreat:
            if id(node) == id(self._cycle_node) or id(node) in self._successful_visits:
                self._cycle_node = None
                self._on_cycle_retreat = False
            else:
                return Discard

    def _collapse_ambig(self, children):
        if False:
            print('Hello World!')
        new_children = []
        for child in children:
            if hasattr(child, 'data') and child.data == '_ambig':
                new_children += child.children
            else:
                new_children.append(child)
        return new_children

    def _call_rule_func(self, node, data):
        if False:
            print('Hello World!')
        return self.callbacks[node.rule](data)

    def _call_ambig_func(self, node, data):
        if False:
            return 10
        if len(data) > 1:
            return self.tree_class('_ambig', data)
        elif data:
            return data[0]
        return Discard

    def transform_symbol_node(self, node, data):
        if False:
            while True:
                i = 10
        if id(node) not in self._successful_visits:
            return Discard
        r = self._check_cycle(node)
        if r is Discard:
            return r
        self._successful_visits.remove(id(node))
        data = self._collapse_ambig(data)
        return self._call_ambig_func(node, data)

    def transform_intermediate_node(self, node, data):
        if False:
            return 10
        if id(node) not in self._successful_visits:
            return Discard
        r = self._check_cycle(node)
        if r is Discard:
            return r
        self._successful_visits.remove(id(node))
        if len(data) > 1:
            children = [self.tree_class('_inter', c) for c in data]
            return self.tree_class('_iambig', children)
        return data[0]

    def transform_packed_node(self, node, data):
        if False:
            print('Hello World!')
        r = self._check_cycle(node)
        if r is Discard:
            return r
        if self.resolve_ambiguity and id(node.parent) in self._successful_visits:
            return Discard
        if self._use_cache and id(node) in self._cache:
            return self._cache[id(node)]
        children = []
        assert len(data) <= 2
        data = PackedData(node, data)
        if data.left is not PackedData.NO_DATA:
            if node.left.is_intermediate and isinstance(data.left, list):
                children += data.left
            else:
                children.append(data.left)
        if data.right is not PackedData.NO_DATA:
            children.append(data.right)
        if node.parent.is_intermediate:
            return self._cache.setdefault(id(node), children)
        return self._cache.setdefault(id(node), self._call_rule_func(node, children))

    def visit_symbol_node_in(self, node):
        if False:
            for i in range(10):
                print('nop')
        super(ForestToParseTree, self).visit_symbol_node_in(node)
        if self._on_cycle_retreat:
            return
        return node.children

    def visit_packed_node_in(self, node):
        if False:
            i = 10
            return i + 15
        self._on_cycle_retreat = False
        to_visit = super(ForestToParseTree, self).visit_packed_node_in(node)
        if not self.resolve_ambiguity or id(node.parent) not in self._successful_visits:
            if not self._use_cache or id(node) not in self._cache:
                return to_visit

    def visit_packed_node_out(self, node):
        if False:
            while True:
                i = 10
        super(ForestToParseTree, self).visit_packed_node_out(node)
        if not self._on_cycle_retreat:
            self._successful_visits.add(id(node.parent))

def handles_ambiguity(func):
    if False:
        return 10
    'Decorator for methods of subclasses of ``TreeForestTransformer``.\n    Denotes that the method should receive a list of transformed derivations.'
    func.handles_ambiguity = True
    return func

class TreeForestTransformer(ForestToParseTree):
    """A ``ForestTransformer`` with a tree ``Transformer``-like interface.
    By default, it will construct a tree.

    Methods provided via inheritance are called based on the rule/symbol
    names of nodes in the forest.

    Methods that act on rules will receive a list of the results of the
    transformations of the rule's children. By default, trees and tokens.

    Methods that act on tokens will receive a token.

    Alternatively, methods that act on rules may be annotated with
    ``handles_ambiguity``. In this case, the function will receive a list
    of all the transformations of all the derivations of the rule.
    By default, a list of trees where each tree.data is equal to the
    rule name or one of its aliases.

    Non-tree transformations are made possible by override of
    ``__default__``, ``__default_token__``, and ``__default_ambig__``.

    Note:
        Tree shaping features such as inlined rules and token filtering are
        not built into the transformation. Positions are also not propagated.

    Parameters:
        tree_class: The tree class to use for construction
        prioritizer: A ``ForestVisitor`` that manipulates the priorities of nodes in the SPPF.
        resolve_ambiguity: If True, ambiguities will be resolved based on priorities.
        use_cache (bool): If True, caches the results of some transformations,
                          potentially improving performance when ``resolve_ambiguity==False``.
                          Only use if you know what you are doing: i.e. All transformation
                          functions are pure and referentially transparent.
    """

    def __init__(self, tree_class=Tree, prioritizer=ForestSumVisitor(), resolve_ambiguity=True, use_cache=False):
        if False:
            for i in range(10):
                print('nop')
        super(TreeForestTransformer, self).__init__(tree_class, dict(), prioritizer, resolve_ambiguity, use_cache)

    def __default__(self, name, data):
        if False:
            print('Hello World!')
        'Default operation on tree (for override).\n\n        Returns a tree with name with data as children.\n        '
        return self.tree_class(name, data)

    def __default_ambig__(self, name, data):
        if False:
            while True:
                i = 10
        "Default operation on ambiguous rule (for override).\n\n        Wraps data in an '_ambig_' node if it contains more than\n        one element.\n        "
        if len(data) > 1:
            return self.tree_class('_ambig', data)
        elif data:
            return data[0]
        return Discard

    def __default_token__(self, node):
        if False:
            print('Hello World!')
        'Default operation on ``Token`` (for override).\n\n        Returns ``node``.\n        '
        return node

    def transform_token_node(self, node):
        if False:
            print('Hello World!')
        return getattr(self, node.type, self.__default_token__)(node)

    def _call_rule_func(self, node, data):
        if False:
            print('Hello World!')
        name = node.rule.alias or node.rule.options.template_source or node.rule.origin.name
        user_func = getattr(self, name, self.__default__)
        if user_func == self.__default__ or hasattr(user_func, 'handles_ambiguity'):
            user_func = partial(self.__default__, name)
        if not self.resolve_ambiguity:
            wrapper = partial(AmbiguousIntermediateExpander, self.tree_class)
            user_func = wrapper(user_func)
        return user_func(data)

    def _call_ambig_func(self, node, data):
        if False:
            print('Hello World!')
        name = node.s.name
        user_func = getattr(self, name, self.__default_ambig__)
        if user_func == self.__default_ambig__ or not hasattr(user_func, 'handles_ambiguity'):
            user_func = partial(self.__default_ambig__, name)
        return user_func(data)

class ForestToPyDotVisitor(ForestVisitor):
    """
    A Forest visitor which writes the SPPF to a PNG.

    The SPPF can get really large, really quickly because
    of the amount of meta-data it stores, so this is probably
    only useful for trivial trees and learning how the SPPF
    is structured.
    """

    def __init__(self, rankdir='TB'):
        if False:
            while True:
                i = 10
        super(ForestToPyDotVisitor, self).__init__(single_visit=True)
        self.pydot = import_module('pydot')
        self.graph = self.pydot.Dot(graph_type='digraph', rankdir=rankdir)

    def visit(self, root, filename):
        if False:
            while True:
                i = 10
        super(ForestToPyDotVisitor, self).visit(root)
        try:
            self.graph.write_png(filename)
        except FileNotFoundError as e:
            logger.error('Could not write png: ', e)

    def visit_token_node(self, node):
        if False:
            i = 10
            return i + 15
        graph_node_id = str(id(node))
        graph_node_label = '"{}"'.format(node.value.replace('"', '\\"'))
        graph_node_color = 8421504
        graph_node_style = '"filled,rounded"'
        graph_node_shape = 'diamond'
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor='#{:06x}'.format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)

    def visit_packed_node_in(self, node):
        if False:
            i = 10
            return i + 15
        graph_node_id = str(id(node))
        graph_node_label = repr(node)
        graph_node_color = 8421504
        graph_node_style = 'filled'
        graph_node_shape = 'diamond'
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor='#{:06x}'.format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)
        yield node.left
        yield node.right

    def visit_packed_node_out(self, node):
        if False:
            for i in range(10):
                print('nop')
        graph_node_id = str(id(node))
        graph_node = self.graph.get_node(graph_node_id)[0]
        for child in [node.left, node.right]:
            if child is not None:
                child_graph_node_id = str(id(child.token if isinstance(child, TokenNode) else child))
                child_graph_node = self.graph.get_node(child_graph_node_id)[0]
                self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node))
            else:
                child_graph_node_id = str(randint(100000000000000000000000000000, 123456789012345678901234567890))
                child_graph_node_style = 'invis'
                child_graph_node = self.pydot.Node(child_graph_node_id, style=child_graph_node_style, label='None')
                child_edge_style = 'invis'
                self.graph.add_node(child_graph_node)
                self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node, style=child_edge_style))

    def visit_symbol_node_in(self, node):
        if False:
            print('Hello World!')
        graph_node_id = str(id(node))
        graph_node_label = repr(node)
        graph_node_color = 8421504
        graph_node_style = '"filled"'
        if node.is_intermediate:
            graph_node_shape = 'ellipse'
        else:
            graph_node_shape = 'rectangle'
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor='#{:06x}'.format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)
        return iter(node.children)

    def visit_symbol_node_out(self, node):
        if False:
            return 10
        graph_node_id = str(id(node))
        graph_node = self.graph.get_node(graph_node_id)[0]
        for child in node.children:
            child_graph_node_id = str(id(child))
            child_graph_node = self.graph.get_node(child_graph_node_id)[0]
            self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node))