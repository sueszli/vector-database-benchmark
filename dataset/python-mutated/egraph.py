from __future__ import annotations
import collections
import itertools
import math
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import Any, TypeVar
from ibis.common.graph import Node
from ibis.util import promote_list
K = TypeVar('K', bound=Hashable)

class DisjointSet(Mapping[K, set[K]]):
    """Disjoint set data structure.

    Also known as union-find data structure. It is a data structure that keeps
    track of a set of elements partitioned into a number of disjoint (non-overlapping)
    subsets. It provides near-constant-time operations to add new sets, to merge
    existing sets, and to determine whether elements are in the same set.

    Parameters
    ----------
    data :
        Initial data to add to the disjoint set.

    Examples
    --------
    >>> ds = DisjointSet()
    >>> ds.add(1)
    1
    >>> ds.add(2)
    2
    >>> ds.add(3)
    3
    >>> ds.union(1, 2)
    True
    >>> ds.union(2, 3)
    True
    >>> ds.find(1)
    1
    >>> ds.find(2)
    1
    >>> ds.find(3)
    1
    >>> ds.union(1, 3)
    False
    """
    __slots__ = ('_parents', '_classes')
    _parents: dict
    _classes: dict

    def __init__(self, data: Iterable[K] | None=None):
        if False:
            print('Hello World!')
        self._parents = {}
        self._classes = {}
        if data is not None:
            for id in data:
                self.add(id)

    def __contains__(self, id) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if the given id is in the disjoint set.\n\n        Parameters\n        ----------\n        id :\n            The id to check.\n\n        Returns\n        -------\n        ined:\n            True if the id is in the disjoint set, False otherwise.\n        '
        return id in self._parents

    def __getitem__(self, id) -> set[K]:
        if False:
            for i in range(10):
                print('nop')
        'Get the set of ids that are in the same class as the given id.\n\n        Parameters\n        ----------\n        id :\n            The id to get the class for.\n\n        Returns\n        -------\n        class:\n            The set of ids that are in the same class as the given id, including\n            the given id.\n        '
        id = self._parents[id]
        return self._classes[id]

    def __iter__(self) -> Iterator[K]:
        if False:
            for i in range(10):
                print('nop')
        'Iterate over the ids in the disjoint set.'
        return iter(self._parents)

    def __len__(self) -> int:
        if False:
            return 10
        'Get the number of ids in the disjoint set.'
        return len(self._parents)

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        'Check if the disjoint set is equal to another disjoint set.\n\n        Parameters\n        ----------\n        other :\n            The other disjoint set to compare to.\n\n        Returns\n        -------\n        equal:\n            True if the disjoint sets are equal, False otherwise.\n        '
        if not isinstance(other, DisjointSet):
            return NotImplemented
        return self._parents == other._parents

    def add(self, id: K) -> K:
        if False:
            return 10
        'Add a new id to the disjoint set.\n\n        If the id is not in the disjoint set, it will be added to the disjoint set\n        along with a new class containing only the given id.\n\n        Parameters\n        ----------\n        id :\n            The id to add to the disjoint set.\n\n        Returns\n        -------\n        id:\n            The id that was added to the disjoint set.\n        '
        if id in self._parents:
            return self._parents[id]
        self._parents[id] = id
        self._classes[id] = {id}
        return id

    def find(self, id: K) -> K:
        if False:
            print('Hello World!')
        'Find the root of the class that the given id is in.\n\n        Also called as the canonicalized id or the representative id.\n\n        Parameters\n        ----------\n        id :\n            The id to find the canonicalized id for.\n\n        Returns\n        -------\n        id:\n            The canonicalized id for the given id.\n        '
        return self._parents[id]

    def union(self, id1, id2) -> bool:
        if False:
            while True:
                i = 10
        'Merge the classes that the given ids are in.\n\n        If the ids are already in the same class, this will return False. Otherwise\n        it will merge the classes and return True.\n\n        Parameters\n        ----------\n        id1 :\n            The first id to merge the classes for.\n        id2 :\n            The second id to merge the classes for.\n\n        Returns\n        -------\n        merged:\n            True if the classes were merged, False otherwise.\n        '
        id1 = self._parents[id1]
        id2 = self._parents[id2]
        if id1 == id2:
            return False
        class1 = self._classes[id1]
        class2 = self._classes[id2]
        if len(class1) >= len(class2):
            (id1, id2) = (id2, id1)
            (class1, class2) = (class2, class1)
        for id in class1:
            self._parents[id] = id2
        class2 |= class1
        class1.clear()
        return True

    def connected(self, id1, id2):
        if False:
            print('Hello World!')
        'Check if the given ids are in the same class.\n\n        True if both ids have the same canonicalized id, False otherwise.\n\n        Parameters\n        ----------\n        id1 :\n            The first id to check.\n        id2 :\n            The second id to check.\n\n        Returns\n        -------\n        connected:\n            True if the ids are connected, False otherwise.\n        '
        return self._parents[id1] == self._parents[id2]

    def verify(self):
        if False:
            for i in range(10):
                print('nop')
        "Verify that the disjoint set is not corrupted.\n\n        Check that each id's canonicalized id's class. In general corruption\n        should not happen if the public API is used, but this is a sanity check\n        to make sure that the internal data structures are not corrupted.\n\n        Returns\n        -------\n        verified:\n            True if the disjoint set is not corrupted, False otherwise.\n        "
        for id in self._parents:
            if id not in self._classes[self._parents[id]]:
                raise RuntimeError(f'DisjointSet is corrupted: {id} is not in its class')

class Slotted:
    """A lightweight alternative to `ibis.common.grounds.Concrete`.

    This class is used to create immutable dataclasses with slots and a precomputed
    hash value for quicker dictionary lookups.
    """
    __slots__ = ('__precomputed_hash__',)
    __precomputed_hash__: int

    def __init__(self, *args):
        if False:
            return 10
        for (name, value) in itertools.zip_longest(self.__slots__, args):
            object.__setattr__(self, name, value)
        object.__setattr__(self, '__precomputed_hash__', hash(args))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        for name in self.__slots__:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self):
        if False:
            print('Hello World!')
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        raise AttributeError("Can't set attributes on immutable ENode instance")

class Variable(Slotted):
    """A named capture in a pattern.

    Parameters
    ----------
    name : str
        The name of the variable.
    """
    __slots__ = ('name',)
    name: str

    def __init__(self, name: str):
        if False:
            i = 10
            return i + 15
        if name is None:
            raise ValueError('Variable name cannot be None')
        super().__init__(name)

    def __repr__(self):
        if False:
            return 10
        return f'${self.name}'

    def substitute(self, egraph, enode, subst):
        if False:
            return 10
        'Substitute the variable with the corresponding value in the substitution.\n\n        Parameters\n        ----------\n        egraph : EGraph\n            The egraph instance.\n        enode : ENode\n            The matched enode.\n        subst : dict\n            The substitution dictionary.\n\n        Returns\n        -------\n        value : Any\n            The substituted value.\n        '
        return subst[self.name]

class Pattern(Slotted):
    """A non-ground term, tree of enodes possibly containing variables.

    This class is used to represent a pattern in a query. The pattern is almost
    identical to an ENode, except that it can contain variables.

    Parameters
    ----------
    head : type
        The head or python type of the ENode to match against.
    args : tuple
        The arguments of the pattern. The arguments can be enodes, patterns,
        variables or leaf values.
    name : str, optional
        The name of the pattern which is used to refer to it in a rewrite rule.
    """
    __slots__ = ('head', 'args', 'name')
    head: type
    args: tuple
    name: str | None

    def __init__(self, head, args, name=None, conditions=None):
        if False:
            return 10
        assert all((not isinstance(arg, (ENode, Node)) for arg in args))
        super().__init__(head, tuple(args), name)

    def matches_none(self):
        if False:
            print('Hello World!')
        'Evaluate whether the pattern is guaranteed to match nothing.\n\n        This can be evaluated before the matching loop starts, so eventually can\n        be eliminated from the flattened query.\n        '
        return len(self.head.__argnames__) != len(self.args)

    def matches_all(self):
        if False:
            return 10
        'Evaluate whether the pattern is guaranteed to match everything.\n\n        This can be evaluated before the matching loop starts, so eventually can\n        be eliminated from the flattened query.\n        '
        return not self.matches_none() and all((isinstance(arg, Variable) for arg in self.args))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        argstring = ', '.join(map(repr, self.args))
        return f'P{self.head.__name__}({argstring})'

    def __rshift__(self, rhs):
        if False:
            return 10
        'Syntax sugar to create a rewrite rule.'
        return Rewrite(self, rhs)

    def __rmatmul__(self, name):
        if False:
            return 10
        'Syntax sugar to create a named pattern.'
        return self.__class__(self.head, self.args, name)

    def flatten(self, var=None, counter=None):
        if False:
            for i in range(10):
                print('nop')
        'Recursively flatten the pattern to a join of selections.\n\n        `Pattern(Add, (Pattern(Mul, ($x, 1)), $y))` is turned into a join of\n        selections by introducing auxiliary variables where each selection gets\n        executed as a dictionary lookup.\n\n        In SQL terms this is equivalent to the following query:\n        SELECT m.0 AS $x, a.1 AS $y FROM Add a JOIN Mul m ON a.0 = m.id WHERE m.1 = 1\n\n        Parameters\n        ----------\n        var : Variable\n            The variable to assign to the flattened pattern.\n        counter : Iterator[int]\n            The counter to generate unique variable names for auxiliary variables\n            connecting the selections.\n\n        Yields\n        ------\n        (var, pattern) : tuple[Variable, Pattern]\n            The variable and the flattened pattern where the flattened pattern\n            cannot contain any patterns just variables.\n        '
        counter = counter or itertools.count()
        if var is None:
            if self.name is None:
                var = Variable(next(counter))
            else:
                var = Variable(self.name)
        args = []
        for arg in self.args:
            if isinstance(arg, Pattern):
                if arg.name is None:
                    aux = Variable(next(counter))
                else:
                    aux = Variable(arg.name)
                yield from arg.flatten(aux, counter)
                args.append(aux)
            else:
                args.append(arg)
        yield (var, Pattern(self.head, args))

    def substitute(self, egraph, enode, subst):
        if False:
            print('Hello World!')
        'Substitute the variables in the pattern with the corresponding values.\n\n        Parameters\n        ----------\n        egraph : EGraph\n            The egraph instance.\n        enode : ENode\n            The matched enode.\n        subst : dict\n            The substitution dictionary.\n\n        Returns\n        -------\n        enode : ENode\n            The substituted pattern which is a ground term aka. an ENode.\n        '
        args = []
        for arg in self.args:
            if isinstance(arg, (Variable, Pattern)):
                arg = arg.substitute(egraph, enode, subst)
            args.append(arg)
        return ENode(self.head, tuple(args))

class DynamicApplier(Slotted):
    """A dynamic applier which calls a function to compute the result."""
    __slots__ = ('func',)
    func: Callable

    def substitute(self, egraph, enode, subst):
        if False:
            print('Hello World!')
        kwargs = {k: v for (k, v) in subst.items() if isinstance(k, str)}
        result = self.func(egraph, enode, **kwargs)
        if not isinstance(result, ENode):
            raise TypeError(f'applier must return an ENode, got {type(result)}')
        return result

class Rewrite(Slotted):
    """A rewrite rule which matches a pattern and applies a pattern or a function."""
    __slots__ = ('matcher', 'applier')
    matcher: Pattern
    applier: Callable | Pattern | Variable

    def __init__(self, matcher, applier):
        if False:
            for i in range(10):
                print('nop')
        if callable(applier):
            applier = DynamicApplier(applier)
        elif not isinstance(applier, (Pattern, Variable)):
            raise TypeError('applier must be a Pattern or a Variable returning an ENode')
        super().__init__(matcher, applier)

    def __repr__(self):
        if False:
            return 10
        return f'{self.lhs} >> {self.rhs}'

class ENode(Slotted, Node):
    """A ground term which is a node in the EGraph, called ENode.

    Parameters
    ----------
    head : type
        The type of the Node the ENode represents.
    args : tuple
        The arguments of the ENode which are either ENodes or leaf values.
    """
    __slots__ = ('head', 'args')
    head: type
    args: tuple

    def __init__(self, head, args):
        if False:
            for i in range(10):
                print('nop')
        assert all((not isinstance(arg, (Pattern, Variable)) for arg in args))
        super().__init__(head, tuple(args))

    @property
    def __argnames__(self):
        if False:
            i = 10
            return i + 15
        'Implementation for the `ibis.common.graph.Node` protocol.'
        return self.head.__argnames__

    @property
    def __args__(self):
        if False:
            print('Hello World!')
        'Implementation for the `ibis.common.graph.Node` protocol.'
        return self.args

    def __repr__(self):
        if False:
            while True:
                i = 10
        argstring = ', '.join(map(repr, self.args))
        return f'E{self.head.__name__}({argstring})'

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return False

    @classmethod
    def from_node(cls, node: Any):
        if False:
            while True:
                i = 10
        'Convert an `ibis.common.graph.Node` to an `ENode`.'

        def mapper(node, _, **kwargs):
            if False:
                return 10
            return cls(node.__class__, kwargs.values())
        return node.map(mapper)[node]

    def to_node(self):
        if False:
            while True:
                i = 10
        'Convert the ENode back to an `ibis.common.graph.Node`.'

        def mapper(node, _, **kwargs):
            if False:
                return 10
            return node.head(**kwargs)
        return self.map(mapper)[self]

class EGraph:
    __slots__ = ('_nodes', '_etables', '_eclasses')
    _nodes: dict
    _etables: collections.defaultdict
    _eclasses: DisjointSet

    def __init__(self):
        if False:
            return 10
        self._nodes = {}
        self._etables = collections.defaultdict(dict)
        self._eclasses = DisjointSet()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'EGraph({self._eclasses})'

    def _as_enode(self, node: Node) -> ENode:
        if False:
            print('Hello World!')
        'Convert a node to an enode.'
        if isinstance(node, ENode):
            return node
        elif isinstance(node, Node):
            return self._nodes.get(node) or ENode.from_node(node)
        else:
            raise TypeError(node)

    def add(self, node: Node) -> ENode:
        if False:
            for i in range(10):
                print('nop')
        'Add a node to the egraph.\n\n        The node is converted to an enode and added to the egraph. If the enode is\n        already present in the egraph, then the canonical enode is returned.\n\n        Parameters\n        ----------\n        node :\n            The node to add to the egraph.\n\n        Returns\n        -------\n        enode :\n            The canonical enode.\n        '
        enode = self._as_enode(node)
        if enode in self._eclasses:
            return self._eclasses.find(enode)
        args = []
        for arg in enode.args:
            if isinstance(arg, ENode):
                args.append(self.add(arg))
            else:
                args.append(arg)
        enode = ENode(enode.head, args)
        self._eclasses.add(enode)
        self._etables[enode.head][enode] = tuple(args)
        return enode

    def union(self, node1: Node, node2: Node) -> ENode:
        if False:
            i = 10
            return i + 15
        'Union two nodes in the egraph.\n\n        The nodes are converted to enodes which must be present in the egraph.\n        The eclasses of the nodes are merged and the canonical enode is returned.\n\n        Parameters\n        ----------\n        node1 :\n            The first node to union.\n        node2 :\n            The second node to union.\n\n        Returns\n        -------\n        enode :\n            The canonical enode.\n        '
        enode1 = self._as_enode(node1)
        enode2 = self._as_enode(node2)
        return self._eclasses.union(enode1, enode2)

    def _match_args(self, args, patargs):
        if False:
            print('Hello World!')
        "Match the arguments of an enode against a pattern's arguments.\n\n        An enode matches a pattern if each of the arguments are:\n        - both leaf values and equal\n        - both enodes and in the same eclass\n        - an enode and a variable, in which case the variable gets bound to the enode\n\n        Parameters\n        ----------\n        args : tuple\n            The arguments of the enode. Since an enode is a ground term, the arguments\n            are either enodes or leaf values.\n        patargs : tuple\n            The arguments of the pattern. Since a pattern is a flat term (flattened\n            using auxiliary variables), the arguments are either variables or leaf\n            values.\n\n        Returns\n        -------\n        dict[str, Any] :\n            The mapping of variable names to enodes or leaf values.\n        "
        subst = {}
        for (arg, patarg) in zip(args, patargs):
            if isinstance(patarg, Variable):
                if isinstance(arg, ENode):
                    subst[patarg.name] = self._eclasses.find(arg)
                else:
                    subst[patarg.name] = arg
            elif patarg != arg:
                return None
        return subst

    def match(self, pattern: Pattern) -> dict[ENode, dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Match a pattern in the egraph.\n\n        The pattern is converted to a conjunctive query (list of flat patterns) and\n        matched against the relations represented by the egraph. This is called the\n        relational e-matching.\n\n        Parameters\n        ----------\n        pattern :\n            The pattern to match in the egraph.\n\n        Returns\n        -------\n        matches :\n            A dictionary mapping the matched enodes to their substitutions.\n        '
        patterns = dict(reversed(list(pattern.flatten())))
        if any((pat.matches_none() for pat in patterns.values())):
            return {}
        ((auxvar, pattern), *rest) = patterns.items()
        matches = {}
        rel = self._etables[pattern.head]
        for (enode, args) in rel.items():
            if (subst := self._match_args(args, pattern.args)) is not None:
                subst[auxvar.name] = enode
                matches[enode] = subst
        for (auxvar, pattern) in rest:
            rel = self._etables[pattern.head]
            tmp = {}
            for (enode, subst) in matches.items():
                if (args := rel.get(subst[auxvar.name])):
                    if (newsubst := self._match_args(args, pattern.args)) is not None:
                        tmp[enode] = {**subst, **newsubst}
            matches = tmp
        return matches

    def apply(self, rewrites: list[Rewrite]) -> int:
        if False:
            print('Hello World!')
        'Apply the given rewrites to the egraph.\n\n        Iteratively match the patterns and apply the rewrites to the graph. The returned\n        number of changes is the number of eclasses that were merged. This is the\n        number of changes made to the egraph. The egraph is saturated if the number of\n        changes is zero.\n\n        Parameters\n        ----------\n        rewrites :\n            A list of rewrites to apply.\n\n        Returns\n        -------\n        n_changes\n            The number of changes made to the egraph.\n        '
        n_changes = 0
        for rewrite in promote_list(rewrites):
            for (match, subst) in self.match(rewrite.matcher).items():
                enode = rewrite.applier.substitute(self, match, subst)
                enode = self.add(enode)
                n_changes += self._eclasses.union(match, enode)
        return n_changes

    def run(self, rewrites: list[Rewrite], n: int=10) -> bool:
        if False:
            return 10
        'Run the match-apply cycles for the given number of iterations.\n\n        Parameters\n        ----------\n        rewrites :\n            A list of rewrites to apply.\n        n :\n            The number of iterations to run.\n\n        Returns\n        -------\n        saturated :\n            True if the egraph is saturated, False otherwise.\n        '
        return any((not self.apply(rewrites) for _i in range(n)))

    def extract(self, node: Node) -> Node:
        if False:
            print('Hello World!')
        'Extract a node from the egraph.\n\n        The node is converted to an enode which recursively gets converted to an\n        enode having the lowest cost according to equivalence classes. Currently\n        the cost function is hardcoded as the depth of the enode.\n\n        Parameters\n        ----------\n        node :\n            The node to extract from the egraph.\n\n        Returns\n        -------\n        node :\n            The extracted node.\n        '
        enode = self._as_enode(node)
        enode = self._eclasses.find(enode)
        costs = {en: (math.inf, None) for en in self._eclasses.keys()}

        def enode_cost(enode):
            if False:
                while True:
                    i = 10
            cost = 1
            for arg in enode.args:
                if isinstance(arg, ENode):
                    cost += costs[arg][0]
                else:
                    cost += 1
            return cost
        changed = True
        while changed:
            changed = False
            for (en, enodes) in self._eclasses.items():
                new_cost = min(((enode_cost(en), en) for en in enodes))
                if costs[en][0] != new_cost[0]:
                    changed = True
                costs[en] = new_cost

        def extract(en):
            if False:
                return 10
            if not isinstance(en, ENode):
                return en
            best = costs[en][1]
            args = tuple((extract(a) for a in best.args))
            return best.head(*args)
        return extract(enode)

    def equivalent(self, node1: Node, node2: Node) -> bool:
        if False:
            return 10
        'Check if two nodes are equivalent.\n\n        The nodes are converted to enodes and checked for equivalence: they are\n        equivalent if they are in the same equivalence class.\n\n        Parameters\n        ----------\n        node1 :\n            The first node.\n        node2 :\n            The second node.\n\n        Returns\n        -------\n        equivalent :\n            True if the nodes are equivalent, False otherwise.\n        '
        enode1 = self._as_enode(node1)
        enode2 = self._as_enode(node2)
        enode1 = self._eclasses.find(enode1)
        enode2 = self._eclasses.find(enode2)
        return enode1 == enode2