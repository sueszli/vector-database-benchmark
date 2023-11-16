"""Base classes for Bio.Phylo objects.

All object representations for phylogenetic trees should derive from these base
classes in order to use the common methods defined on them.
"""
import collections
import copy
import itertools
import random
import re
import warnings

def _level_traverse(root, get_children):
    if False:
        i = 10
        return i + 15
    'Traverse a tree in breadth-first (level) order (PRIVATE).'
    Q = collections.deque([root])
    while Q:
        v = Q.popleft()
        yield v
        Q.extend(get_children(v))

def _preorder_traverse(root, get_children):
    if False:
        print('Hello World!')
    'Traverse a tree in depth-first pre-order (parent before children) (PRIVATE).'

    def dfs(elem):
        if False:
            i = 10
            return i + 15
        yield elem
        for v in get_children(elem):
            yield from dfs(v)
    yield from dfs(root)

def _postorder_traverse(root, get_children):
    if False:
        i = 10
        return i + 15
    'Traverse a tree in depth-first post-order (children before parent) (PRIVATE).'

    def dfs(elem):
        if False:
            for i in range(10):
                print('nop')
        for v in get_children(elem):
            yield from dfs(v)
        yield elem
    yield from dfs(root)

def _sorted_attrs(elem):
    if False:
        for i in range(10):
            print('nop')
    "Get a flat list of elem's attributes, sorted for consistency (PRIVATE)."
    singles = []
    lists = []
    for (attrname, child) in sorted(elem.__dict__.items(), key=lambda kv: kv[0]):
        if child is None:
            continue
        if isinstance(child, list):
            lists.extend(child)
        else:
            singles.append(child)
    return (x for x in singles + lists if isinstance(x, TreeElement))

def _identity_matcher(target):
    if False:
        while True:
            i = 10
    'Match a node to the target object by identity (PRIVATE).'

    def match(node):
        if False:
            for i in range(10):
                print('nop')
        return node is target
    return match

def _class_matcher(target_cls):
    if False:
        return 10
    "Match a node if it's an instance of the given class (PRIVATE)."

    def match(node):
        if False:
            return 10
        return isinstance(node, target_cls)
    return match

def _string_matcher(target):
    if False:
        for i in range(10):
            print('nop')

    def match(node):
        if False:
            i = 10
            return i + 15
        if isinstance(node, (Clade, Tree)):
            return node.name == target
        return str(node) == target
    return match

def _attribute_matcher(kwargs):
    if False:
        while True:
            i = 10
    "Match a node by specified attribute values (PRIVATE).\n\n    ``terminal`` is a special case: True restricts the search to external (leaf)\n    nodes, False restricts to internal nodes, and None allows all tree elements\n    to be searched, including phyloXML annotations.\n\n    Otherwise, for a tree element to match the specification (i.e. for the\n    function produced by ``_attribute_matcher`` to return True when given a tree\n    element), it must have each of the attributes specified by the keys and\n    match each of the corresponding values -- think 'and', not 'or', for\n    multiple keys.\n    "

    def match(node):
        if False:
            while True:
                i = 10
        if 'terminal' in kwargs:
            kwa_copy = kwargs.copy()
            pattern = kwa_copy.pop('terminal')
            if pattern is not None and (not hasattr(node, 'is_terminal') or node.is_terminal() != pattern):
                return False
        else:
            kwa_copy = kwargs
        for (key, pattern) in kwa_copy.items():
            if not hasattr(node, key):
                return False
            target = getattr(node, key)
            if isinstance(pattern, str):
                return isinstance(target, str) and re.match(pattern + '$', target)
            if isinstance(pattern, bool):
                return pattern == bool(target)
            if isinstance(pattern, int):
                return pattern == target
            if pattern is None:
                return target is None
            raise TypeError(f'invalid query type: {type(pattern)}')
        return True
    return match

def _function_matcher(matcher_func):
    if False:
        while True:
            i = 10
    'Safer attribute lookup -- returns False instead of raising an error (PRIVATE).'

    def match(node):
        if False:
            for i in range(10):
                print('nop')
        try:
            return matcher_func(node)
        except (LookupError, AttributeError, ValueError, TypeError):
            return False
    return match

def _object_matcher(obj):
    if False:
        return 10
    "Retrieve a matcher function by passing an arbitrary object (PRIVATE).\n\n    Passing a ``TreeElement`` such as a ``Clade`` or ``Tree`` instance returns\n    an identity matcher, passing a type such as the ``PhyloXML.Taxonomy`` class\n    returns a class matcher, and passing a dictionary returns an attribute\n    matcher.\n\n    The resulting 'match' function returns True when given an object matching\n    the specification (identity, type or attribute values), otherwise False.\n    This is useful for writing functions that search the tree, and probably\n    shouldn't be used directly by the end user.\n    "
    if isinstance(obj, TreeElement):
        return _identity_matcher(obj)
    if isinstance(obj, type):
        return _class_matcher(obj)
    if isinstance(obj, str):
        return _string_matcher(obj)
    if isinstance(obj, dict):
        return _attribute_matcher(obj)
    if callable(obj):
        return _function_matcher(obj)
    raise ValueError(f'{obj} (type {type(obj)}) is not a valid type for comparison.')

def _combine_matchers(target, kwargs, require_spec):
    if False:
        while True:
            i = 10
    'Merge target specifications with keyword arguments (PRIVATE).\n\n    Dispatch the components to the various matcher functions, then merge into a\n    single boolean function.\n    '
    if not target:
        if not kwargs:
            if require_spec:
                raise ValueError('you must specify a target object or keyword arguments.')
            return lambda x: True
        return _attribute_matcher(kwargs)
    match_obj = _object_matcher(target)
    if not kwargs:
        return match_obj
    match_kwargs = _attribute_matcher(kwargs)
    return lambda x: match_obj(x) and match_kwargs(x)

def _combine_args(first, *rest):
    if False:
        for i in range(10):
            print('nop')
    'Convert ``[targets]`` or ``*targets`` arguments to a single iterable (PRIVATE).\n\n    This helps other functions work like the built-in functions ``max`` and\n    ``min``.\n    '
    if hasattr(first, '__iter__') and (not isinstance(first, (TreeElement, dict, str, type))):
        if rest:
            raise ValueError('Arguments must be either a single list of targets, or separately specified targets (e.g. foo(t1, t2, t3)), but not both.')
        return first
    return itertools.chain([first], rest)

class TreeElement:
    """Base class for all Bio.Phylo classes."""

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        "Show this object's constructor with its primitive arguments."

        def pair_as_kwarg_string(key, val):
            if False:
                i = 10
                return i + 15
            if isinstance(val, str):
                val = val[:57] + '...' if len(val) > 60 else val
                return f"{key}='{val}'"
            return f'{key}={val}'
        return '%s(%s)' % (self.__class__.__name__, ', '.join((pair_as_kwarg_string(key, val) for (key, val) in sorted(self.__dict__.items()) if val is not None and type(val) in (str, int, float, bool, str))))

    def __str__(self) -> str:
        if False:
            return 10
        return self.__repr__()

class TreeMixin:
    """Methods for Tree- and Clade-based classes.

    This lets ``Tree`` and ``Clade`` support the same traversal and searching
    operations without requiring Clade to inherit from Tree, so Clade isn't
    required to have all of Tree's attributes -- just ``root`` (a Clade
    instance) and ``is_terminal``.
    """

    def _filter_search(self, filter_func, order, follow_attrs):
        if False:
            return 10
        'Perform a BFS or DFS traversal through all elements in this tree (PRIVATE).\n\n        :returns: generator of all elements for which ``filter_func`` is True.\n\n        '
        order_opts = {'preorder': _preorder_traverse, 'postorder': _postorder_traverse, 'level': _level_traverse}
        try:
            order_func = order_opts[order]
        except KeyError:
            raise ValueError(f"Invalid order '{order}'; must be one of: {tuple(order_opts)}") from None
        if follow_attrs:
            get_children = _sorted_attrs
            root = self
        else:
            get_children = lambda elem: elem.clades
            root = self.root
        return filter(filter_func, order_func(root, get_children))

    def find_any(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Return the first element found by find_elements(), or None.\n\n        This is also useful for checking whether any matching element exists in\n        the tree, and can be used in a conditional expression.\n        '
        hits = self.find_elements(*args, **kwargs)
        try:
            return next(hits)
        except StopIteration:
            return None

    def find_elements(self, target=None, terminal=None, order='preorder', **kwargs):
        if False:
            i = 10
            return i + 15
        "Find all tree elements matching the given attributes.\n\n        The arbitrary keyword arguments indicate the attribute name of the\n        sub-element and the value to match: string, integer or boolean. Strings\n        are evaluated as regular expression matches; integers are compared\n        directly for equality, and booleans evaluate the attribute's truth value\n        (True or False) before comparing. To handle nonzero floats, search with\n        a boolean argument, then filter the result manually.\n\n        If no keyword arguments are given, then just the class type is used for\n        matching.\n\n        The result is an iterable through all matching objects, by depth-first\n        search. (Not necessarily the same order as the elements appear in the\n        source file!)\n\n        :Parameters:\n            target : TreeElement instance, type, dict, or callable\n                Specifies the characteristics to search for. (The default,\n                TreeElement, matches any standard Bio.Phylo type.)\n            terminal : bool\n                A boolean value to select for or against terminal nodes (a.k.a.\n                leaf nodes). True searches for only terminal nodes, False\n                excludes terminal nodes, and the default, None, searches both\n                terminal and non-terminal nodes, as well as any tree elements\n                lacking the ``is_terminal`` method.\n            order : {'preorder', 'postorder', 'level'}\n                Tree traversal order: 'preorder' (default) is depth-first\n                search, 'postorder' is DFS with child nodes preceding parents,\n                and 'level' is breadth-first search.\n\n        Examples\n        --------\n        >>> from Bio import Phylo\n        >>> phx = Phylo.PhyloXMLIO.read('PhyloXML/phyloxml_examples.xml')\n        >>> matches = phx.phylogenies[5].find_elements(code='OCTVU')\n        >>> next(matches)\n        Taxonomy(code='OCTVU', scientific_name='Octopus vulgaris')\n\n        "
        if terminal is not None:
            kwargs['terminal'] = terminal
        is_matching_elem = _combine_matchers(target, kwargs, False)
        return self._filter_search(is_matching_elem, order, True)

    def find_clades(self, target=None, terminal=None, order='preorder', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Find each clade containing a matching element.\n\n        That is, find each element as with find_elements(), but return the\n        corresponding clade object. (This is usually what you want.)\n\n        :returns: an iterable through all matching objects, searching\n            depth-first (preorder) by default.\n\n        '

        def match_attrs(elem):
            if False:
                for i in range(10):
                    print('nop')
            orig_clades = elem.__dict__.pop('clades')
            found = elem.find_any(target, **kwargs)
            elem.clades = orig_clades
            return found is not None
        if terminal is None:
            is_matching_elem = match_attrs
        else:

            def is_matching_elem(elem):
                if False:
                    while True:
                        i = 10
                return elem.is_terminal() == terminal and match_attrs(elem)
        return self._filter_search(is_matching_elem, order, False)

    def get_path(self, target=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'List the clades directly between this root and the given target.\n\n        :returns: list of all clade objects along this path, ending with the\n            given target, but excluding the root clade.\n\n        '
        path = []
        match = _combine_matchers(target, kwargs, True)

        def check_in_path(v):
            if False:
                for i in range(10):
                    print('nop')
            if match(v):
                path.append(v)
                return True
            elif v.is_terminal():
                return False
            for child in v:
                if check_in_path(child):
                    path.append(v)
                    return True
            return False
        if not check_in_path(self.root):
            return None
        return path[-2::-1]

    def get_nonterminals(self, order='preorder'):
        if False:
            i = 10
            return i + 15
        "Get a list of all of this tree's nonterminal (internal) nodes."
        return list(self.find_clades(terminal=False, order=order))

    def get_terminals(self, order='preorder'):
        if False:
            print('Hello World!')
        "Get a list of all of this tree's terminal (leaf) nodes."
        return list(self.find_clades(terminal=True, order=order))

    def trace(self, start, finish):
        if False:
            i = 10
            return i + 15
        'List of all clade object between two targets in this tree.\n\n        Excluding ``start``, including ``finish``.\n        '
        mrca = self.common_ancestor(start, finish)
        fromstart = mrca.get_path(start)[-2::-1]
        to = mrca.get_path(finish)
        return fromstart + [mrca] + to

    def common_ancestor(self, targets, *more_targets):
        if False:
            while True:
                i = 10
        'Most recent common ancestor (clade) of all the given targets.\n\n        Edge cases:\n         - If no target is given, returns self.root\n         - If 1 target is given, returns the target\n         - If any target is not found in this tree, raises a ValueError\n\n        '
        paths = [self.get_path(t) for t in _combine_args(targets, *more_targets)]
        for (p, t) in zip(paths, targets):
            if p is None:
                raise ValueError(f'target {t!r} is not in this tree')
        mrca = self.root
        for level in zip(*paths):
            ref = level[0]
            for other in level[1:]:
                if ref is not other:
                    break
            else:
                mrca = ref
            if ref is not mrca:
                break
        return mrca

    def count_terminals(self):
        if False:
            return 10
        'Count the number of terminal (leaf) nodes within this tree.'
        return sum((1 for clade in self.find_clades(terminal=True)))

    def depths(self, unit_branch_lengths=False):
        if False:
            return 10
        'Create a mapping of tree clades to depths (by branch length).\n\n        :Parameters:\n            unit_branch_lengths : bool\n                If True, count only the number of branches (levels in the tree).\n                By default the distance is the cumulative branch length leading\n                to the clade.\n\n        :returns: dict of {clade: depth}, where keys are all of the Clade\n            instances in the tree, and values are the distance from the root to\n            each clade (including terminals).\n\n        '
        if unit_branch_lengths:
            depth_of = lambda c: 1
        else:
            depth_of = lambda c: c.branch_length or 0
        depths = {}

        def update_depths(node, curr_depth):
            if False:
                for i in range(10):
                    print('nop')
            depths[node] = curr_depth
            for child in node.clades:
                new_depth = curr_depth + depth_of(child)
                update_depths(child, new_depth)
        update_depths(self.root, self.root.branch_length or 0)
        return depths

    def distance(self, target1, target2=None):
        if False:
            print('Hello World!')
        'Calculate the sum of the branch lengths between two targets.\n\n        If only one target is specified, the other is the root of this tree.\n        '
        if target2 is None:
            return sum((n.branch_length for n in self.get_path(target1) if n.branch_length is not None))
        mrca = self.common_ancestor(target1, target2)
        return mrca.distance(target1) + mrca.distance(target2)

    def is_bifurcating(self):
        if False:
            while True:
                i = 10
        'Return True if tree downstream of node is strictly bifurcating.\n\n        I.e., all nodes have either 2 or 0 children (internal or external,\n        respectively). The root may have 3 descendents and still be considered\n        part of a bifurcating tree, because it has no ancestor.\n        '
        if isinstance(self, Tree) and len(self.root) == 3:
            return self.root.clades[0].is_bifurcating() and self.root.clades[1].is_bifurcating() and self.root.clades[2].is_bifurcating()
        if len(self.root) == 2:
            return self.root.clades[0].is_bifurcating() and self.root.clades[1].is_bifurcating()
        if len(self.root) == 0:
            return True
        return False

    def is_monophyletic(self, terminals, *more_terminals):
        if False:
            i = 10
            return i + 15
        'MRCA of terminals if they comprise a complete subclade, or False.\n\n        I.e., there exists a clade such that its terminals are the same set as\n        the given targets.\n\n        The given targets must be terminals of the tree.\n\n        To match both ``Bio.Nexus.Trees`` and the other multi-target methods in\n        Bio.Phylo, arguments to this method can be specified either of two ways:\n        (i) as a single list of targets, or (ii) separately specified targets,\n        e.g. is_monophyletic(t1, t2, t3) -- but not both.\n\n        For convenience, this method returns the common ancestor (MCRA) of the\n        targets if they are monophyletic (instead of the value True), and False\n        otherwise.\n\n        :returns: common ancestor if terminals are monophyletic, otherwise False.\n\n        '
        target_set = set(_combine_args(terminals, *more_terminals))
        current = self.root
        while True:
            if set(current.get_terminals()) == target_set:
                return current
            for subclade in current.clades:
                if set(subclade.get_terminals()).issuperset(target_set):
                    current = subclade
                    break
            else:
                return False

    def is_parent_of(self, target=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Check if target is a descendent of this tree.\n\n        Not required to be a direct descendent.\n\n        To check only direct descendents of a clade, simply use list membership\n        testing: ``if subclade in clade: ...``\n        '
        return self.get_path(target, **kwargs) is not None

    def is_preterminal(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if all direct descendents are terminal.'
        if self.root.is_terminal():
            return False
        for clade in self.root.clades:
            if not clade.is_terminal():
                return False
        return True

    def total_branch_length(self):
        if False:
            return 10
        'Calculate the sum of all the branch lengths in this tree.'
        return sum((node.branch_length for node in self.find_clades(branch_length=True)))

    def collapse(self, target=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Delete target from the tree, relinking its children to its parent.\n\n        :returns: the parent clade.\n\n        '
        path = self.get_path(target, **kwargs)
        if not path:
            raise ValueError("couldn't collapse %s in this tree" % (target or kwargs))
        if len(path) == 1:
            parent = self.root
        else:
            parent = path[-2]
        popped = parent.clades.pop(parent.clades.index(path[-1]))
        extra_length = popped.branch_length or 0
        for child in popped:
            child.branch_length += extra_length
        parent.clades.extend(popped.clades)
        return parent

    def collapse_all(self, target=None, **kwargs):
        if False:
            return 10
        'Collapse all the descendents of this tree, leaving only terminals.\n\n        Total branch lengths are preserved, i.e. the distance to each terminal\n        stays the same.\n\n        For example, this will safely collapse nodes with poor bootstrap\n        support:\n\n            >>> from Bio import Phylo\n            >>> tree = Phylo.read(\'PhyloXML/apaf.xml\', \'phyloxml\')\n            >>> print("Total branch length %0.2f" % tree.total_branch_length())\n            Total branch length 20.44\n            >>> tree.collapse_all(lambda c: c.confidence is not None and c.confidence < 70)\n            >>> print("Total branch length %0.2f" % tree.total_branch_length())\n            Total branch length 21.37\n\n        This implementation avoids strange side-effects by using level-order\n        traversal and testing all clade properties (versus the target\n        specification) up front. In particular, if a clade meets the target\n        specification in the original tree, it will be collapsed.  For example,\n        if the condition is:\n\n            >>> from Bio import Phylo\n            >>> tree = Phylo.read(\'PhyloXML/apaf.xml\', \'phyloxml\')\n            >>> print("Total branch length %0.2f" % tree.total_branch_length())\n            Total branch length 20.44\n            >>> tree.collapse_all(lambda c: c.branch_length < 0.1)\n            >>> print("Total branch length %0.2f" % tree.total_branch_length())\n            Total branch length 21.13\n\n        Collapsing a clade\'s parent node adds the parent\'s branch length to the\n        child, so during the execution of collapse_all, a clade\'s branch_length\n        may increase. In this implementation, clades are collapsed according to\n        their properties in the original tree, not the properties when tree\n        traversal reaches the clade. (It\'s easier to debug.) If you want the\n        other behavior (incremental testing), modifying the source code of this\n        function is straightforward.\n        '
        matches = list(self.find_clades(target, False, 'level', **kwargs))
        if not matches:
            return
        if matches[0] == self.root:
            matches.pop(0)
        for clade in matches:
            self.collapse(clade)

    def ladderize(self, reverse=False):
        if False:
            print('Hello World!')
        'Sort clades in-place according to the number of terminal nodes.\n\n        Deepest clades are last by default. Use ``reverse=True`` to sort clades\n        deepest-to-shallowest.\n        '
        self.root.clades.sort(key=lambda c: c.count_terminals(), reverse=reverse)
        for subclade in self.root.clades:
            subclade.ladderize(reverse=reverse)

    def prune(self, target=None, **kwargs):
        if False:
            return 10
        'Prunes a terminal clade from the tree.\n\n        If taxon is from a bifurcation, the connecting node will be collapsed\n        and its branch length added to remaining terminal node. This might be no\n        longer be a meaningful value.\n\n        :returns: parent clade of the pruned target\n\n        '
        if 'terminal' in kwargs and kwargs['terminal']:
            raise ValueError('target must be terminal')
        path = self.get_path(target, terminal=True, **kwargs)
        if not path:
            raise ValueError("can't find a matching target below this root")
        if len(path) == 1:
            parent = self.root
        else:
            parent = path[-2]
        parent.clades.remove(path[-1])
        if len(parent) == 1:
            if parent == self.root:
                newroot = parent.clades[0]
                newroot.branch_length = None
                parent = self.root = newroot
            else:
                child = parent.clades[0]
                if child.branch_length is not None:
                    child.branch_length += parent.branch_length or 0.0
                if len(path) < 3:
                    grandparent = self.root
                else:
                    grandparent = path[-3]
                index = grandparent.clades.index(parent)
                grandparent.clades.pop(index)
                grandparent.clades.insert(index, child)
                parent = grandparent
        return parent

    def split(self, n=2, branch_length=1.0):
        if False:
            for i in range(10):
                print('nop')
        'Generate n (default 2) new descendants.\n\n        In a species tree, this is a speciation event.\n\n        New clades have the given branch_length and the same name as this\n        clade\'s root plus an integer suffix (counting from 0). For example,\n        splitting a clade named "A" produces sub-clades named "A0" and "A1".\n        If the clade has no name, the prefix "n" is used for child nodes, e.g.\n        "n0" and "n1".\n        '
        clade_cls = type(self.root)
        base_name = self.root.name or 'n'
        for i in range(n):
            clade = clade_cls(name=base_name + str(i), branch_length=branch_length)
            self.root.clades.append(clade)

class Tree(TreeElement, TreeMixin):
    """A phylogenetic tree, containing global info for the phylogeny.

    The structure and node-specific data is accessible through the 'root'
    clade attached to the Tree instance.

    :Parameters:
        root : Clade
            The starting node of the tree. If the tree is rooted, this will
            usually be the root node.
        rooted : bool
            Whether or not the tree is rooted. By default, a tree is assumed to
            be rooted.
        id : str
            The identifier of the tree, if there is one.
        name : str
            The name of the tree, in essence a label.

    """

    def __init__(self, root=None, rooted=True, id=None, name=None):
        if False:
            print('Hello World!')
        'Initialize parameter for phylogenetic tree.'
        self.root = root or Clade()
        self.rooted = rooted
        self.id = id
        self.name = name

    @classmethod
    def from_clade(cls, clade, **kwargs):
        if False:
            return 10
        'Create a new Tree object given a clade.\n\n        Keyword arguments are the usual ``Tree`` constructor parameters.\n        '
        root = copy.deepcopy(clade)
        return cls(root, **kwargs)

    @classmethod
    def randomized(cls, taxa, branch_length=1.0, branch_stdev=None):
        if False:
            print('Hello World!')
        'Create a randomized bifurcating tree given a list of taxa.\n\n        :param taxa: Either an integer specifying the number of taxa to create\n            (automatically named taxon#), or an iterable of taxon names, as\n            strings.\n\n        :returns: a tree of the same type as this class.\n\n        '
        if isinstance(taxa, int):
            taxa = [f'taxon{i + 1}' for i in range(taxa)]
        elif hasattr(taxa, '__iter__'):
            taxa = list(taxa)
        else:
            raise TypeError('taxa argument must be integer (# taxa) or iterable of taxon names.')
        rtree = cls()
        terminals = [rtree.root]
        while len(terminals) < len(taxa):
            newsplit = random.choice(terminals)
            newsplit.split(branch_length=branch_length)
            newterms = newsplit.clades
            if branch_stdev:
                for nt in newterms:
                    nt.branch_length = max(0, random.gauss(branch_length, branch_stdev))
            terminals.remove(newsplit)
            terminals.extend(newterms)
        random.shuffle(taxa)
        for (node, name) in zip(terminals, taxa):
            node.name = name
        return rtree

    @property
    def clade(self):
        if False:
            while True:
                i = 10
        'Return first clade in this tree (not itself).'
        return self.root

    def as_phyloxml(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Convert this tree to a PhyloXML-compatible Phylogeny.\n\n        This lets you use the additional annotation types PhyloXML defines, and\n        save this information when you write this tree as 'phyloxml'.\n        "
        from Bio.Phylo.PhyloXML import Phylogeny
        return Phylogeny.from_tree(self, **kwargs)

    def root_with_outgroup(self, outgroup_targets, *more_targets, outgroup_branch_length=None):
        if False:
            i = 10
            return i + 15
        'Reroot this tree with the outgroup clade containing outgroup_targets.\n\n        Operates in-place.\n\n        Edge cases:\n         - If ``outgroup == self.root``, no change\n         - If outgroup is terminal, create new bifurcating root node with a\n           0-length branch to the outgroup\n         - If outgroup is internal, use the given outgroup node as the new\n           trifurcating root, keeping branches the same\n         - If the original root was bifurcating, drop it from the tree,\n           preserving total branch lengths\n\n        :param outgroup_branch_length: length of the branch leading to the\n            outgroup after rerooting. If not specified (None), then:\n\n            - If the outgroup is an internal node (not a single terminal taxon),\n              then use that node as the new root.\n            - Otherwise, create a new root node as the parent of the outgroup.\n\n        '
        outgroup = self.common_ancestor(outgroup_targets, *more_targets)
        outgroup_path = self.get_path(outgroup)
        if len(outgroup_path) == 0:
            return
        prev_blen = outgroup.branch_length or 0.0
        if outgroup.is_terminal() or outgroup_branch_length is not None:
            outgroup.branch_length = outgroup_branch_length or 0.0
            new_root = self.root.__class__(branch_length=self.root.branch_length, clades=[outgroup])
            if len(outgroup_path) == 1:
                new_parent = new_root
            else:
                parent = outgroup_path.pop(-2)
                parent.clades.pop(parent.clades.index(outgroup))
                (prev_blen, parent.branch_length) = (parent.branch_length, prev_blen - outgroup.branch_length)
                new_root.clades.insert(0, parent)
                new_parent = parent
        else:
            new_root = outgroup
            new_root.branch_length = self.root.branch_length
            new_parent = new_root
        for parent in outgroup_path[-2::-1]:
            parent.clades.pop(parent.clades.index(new_parent))
            (prev_blen, parent.branch_length) = (parent.branch_length, prev_blen)
            new_parent.clades.insert(0, parent)
            new_parent = parent
        old_root = self.root
        if outgroup in old_root.clades:
            assert len(outgroup_path) == 1
            old_root.clades.pop(old_root.clades.index(outgroup))
        else:
            old_root.clades.pop(old_root.clades.index(new_parent))
        if len(old_root) == 1:
            ingroup = old_root.clades[0]
            if ingroup.branch_length:
                ingroup.branch_length += prev_blen
            else:
                ingroup.branch_length = prev_blen
            new_parent.clades.insert(0, ingroup)
        else:
            old_root.branch_length = prev_blen
            new_parent.clades.insert(0, old_root)
        self.root = new_root
        self.rooted = True

    def root_at_midpoint(self):
        if False:
            return 10
        'Root the tree at the midpoint of the two most distant taxa.\n\n        This operates in-place, leaving a bifurcating root. The topology of the\n        tree is otherwise retained, though no guarantees are made about the\n        stability of clade/node/taxon ordering.\n        '
        max_distance = 0.0
        tips = self.get_terminals()
        for tip in tips:
            self.root_with_outgroup(tip)
            new_max = max(self.depths().items(), key=lambda nd: nd[1])
            if new_max[1] > max_distance:
                tip1 = tip
                tip2 = new_max[0]
                max_distance = new_max[1]
        self.root_with_outgroup(tip1)
        root_remainder = 0.5 * (max_distance - (self.root.branch_length or 0))
        assert root_remainder >= 0
        for node in self.get_path(tip2):
            root_remainder -= node.branch_length
            if root_remainder < 0:
                outgroup_node = node
                outgroup_branch_length = -root_remainder
                break
        else:
            raise ValueError('Somehow, failed to find the midpoint!')
        self.root_with_outgroup(outgroup_node, outgroup_branch_length=outgroup_branch_length)

    def is_terminal(self):
        if False:
            print('Hello World!')
        'Check if the root of this tree is terminal.'
        return not self.root.clades

    def __format__(self, format_spec):
        if False:
            for i in range(10):
                print('nop')
        "Serialize the tree as a string in the specified file format.\n\n        This method supports Python's ``format`` built-in function.\n\n        :param format_spec: a lower-case string supported by ``Bio.Phylo.write``\n            as an output file format.\n\n        "
        if format_spec:
            from io import StringIO
            from Bio.Phylo import _io
            handle = StringIO()
            _io.write([self], handle, format_spec)
            return handle.getvalue()
        else:
            return str(self)

    def format(self, fmt=None):
        if False:
            return 10
        'Serialize the tree as a string in the specified file format.\n\n        :param fmt: a lower-case string supported by ``Bio.Phylo.write``\n            as an output file format.\n\n        '
        return self.__format__(fmt)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return a string representation of the entire tree.\n\n        Serialize each sub-clade recursively using ``repr`` to create a summary\n        of the object structure.\n        '
        TAB = '    '
        textlines = []

        def print_tree(obj, indent):
            if False:
                for i in range(10):
                    print('nop')
            'Recursively serialize sub-elements.\n\n            This closes over textlines and modifies it in-place.\n            '
            if isinstance(obj, (Tree, Clade)):
                objstr = repr(obj)
            else:
                objstr = str(obj)
            textlines.append(TAB * indent + objstr)
            indent += 1
            for attr in obj.__dict__:
                child = getattr(obj, attr)
                if isinstance(child, TreeElement):
                    print_tree(child, indent)
                elif isinstance(child, list):
                    for elem in child:
                        if isinstance(elem, TreeElement):
                            print_tree(elem, indent)
        print_tree(self, 0)
        return '\n'.join(textlines)

class Clade(TreeElement, TreeMixin):
    """A recursively defined sub-tree.

    :Parameters:
        branch_length : str
            The length of the branch leading to the root node of this clade.
        name : str
            The clade's name (a label).
        clades : list
            Sub-trees rooted directly under this tree's root.
        confidence : number
            Support.
        color : BranchColor
            The display color of the branch and descendents.
        width : number
            The display width of the branch and descendents.

    """

    def __init__(self, branch_length=None, name=None, clades=None, confidence=None, color=None, width=None):
        if False:
            for i in range(10):
                print('nop')
        'Define parameters for the Clade tree.'
        self.branch_length = branch_length
        self.name = name
        self.clades = clades or []
        self.confidence = confidence
        self.color = color
        self.width = width

    @property
    def root(self):
        if False:
            for i in range(10):
                print('nop')
        'Allow TreeMixin methods to traverse clades properly.'
        return self

    def is_terminal(self):
        if False:
            print('Hello World!')
        'Check if this is a terminal (leaf) node.'
        return not self.clades

    def __getitem__(self, index):
        if False:
            return 10
        'Get clades by index (integer or slice).'
        if isinstance(index, (int, slice)):
            return self.clades[index]
        ref = self
        for idx in index:
            ref = ref[idx]
        return ref

    def __iter__(self):
        if False:
            while True:
                i = 10
        "Iterate through this tree's direct descendent clades (sub-trees)."
        return iter(self.clades)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of clades directly under the root.'
        return len(self.clades)

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        'Boolean value of an instance of this class (True).\n\n        NB: If this method is not defined, but ``__len__``  is, then the object\n        is considered true if the result of ``__len__()`` is nonzero. We want\n        Clade instances to always be considered True.\n        '
        return True

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        'Return name of the class instance.'
        if self.name:
            return self.name[:37] + '...' if len(self.name) > 40 else self.name
        return self.__class__.__name__

    def _get_color(self):
        if False:
            for i in range(10):
                print('nop')
        return self._color

    def _set_color(self, arg):
        if False:
            for i in range(10):
                print('nop')
        if arg is None or isinstance(arg, BranchColor):
            self._color = arg
        elif isinstance(arg, str):
            if arg in BranchColor.color_names:
                self._color = BranchColor.from_name(arg)
            elif arg.startswith('#') and len(arg) == 7:
                self._color = BranchColor.from_hex(arg)
            else:
                raise ValueError(f'invalid color string {arg}')
        elif hasattr(arg, '__iter__') and len(arg) == 3:
            self._color = BranchColor(*arg)
        else:
            raise ValueError(f'invalid color value {arg}')
    color = property(_get_color, _set_color, doc='Branch color.')

class BranchColor:
    """Indicates the color of a clade when rendered graphically.

    The color should be interpreted by client code (e.g. visualization
    programs) as applying to the whole clade, unless overwritten by the
    color(s) of sub-clades.

    Color values must be integers from 0 to 255.
    """
    color_names = {'red': (255, 0, 0), 'r': (255, 0, 0), 'yellow': (255, 255, 0), 'y': (255, 255, 0), 'green': (0, 128, 0), 'g': (0, 128, 0), 'cyan': (0, 255, 255), 'c': (0, 255, 255), 'blue': (0, 0, 255), 'b': (0, 0, 255), 'magenta': (255, 0, 255), 'm': (255, 0, 255), 'black': (0, 0, 0), 'k': (0, 0, 0), 'white': (255, 255, 255), 'w': (255, 255, 255), 'maroon': (128, 0, 0), 'olive': (128, 128, 0), 'lime': (0, 255, 0), 'aqua': (0, 255, 255), 'teal': (0, 128, 128), 'navy': (0, 0, 128), 'fuchsia': (255, 0, 255), 'purple': (128, 0, 128), 'silver': (192, 192, 192), 'gray': (128, 128, 128), 'grey': (128, 128, 128), 'pink': (255, 192, 203), 'salmon': (250, 128, 114), 'orange': (255, 165, 0), 'gold': (255, 215, 0), 'tan': (210, 180, 140), 'brown': (165, 42, 42)}

    def __init__(self, red, green, blue):
        if False:
            print('Hello World!')
        'Initialize BranchColor for a tree.'
        for color in (red, green, blue):
            assert isinstance(color, int) and 0 <= color <= 255, 'Color values must be integers between 0 and 255.'
        self.red = red
        self.green = green
        self.blue = blue

    @classmethod
    def from_hex(cls, hexstr):
        if False:
            print('Hello World!')
        "Construct a BranchColor object from a hexadecimal string.\n\n        The string format is the same style used in HTML and CSS, such as\n        '#FF8000' for an RGB value of (255, 128, 0).\n        "
        assert isinstance(hexstr, str) and hexstr.startswith('#') and (len(hexstr) == 7), 'need a 24-bit hexadecimal string, e.g. #000000'
        RGB = (hexstr[1:3], hexstr[3:5], hexstr[5:])
        return cls(*(int('0x' + cc, base=16) for cc in RGB))

    @classmethod
    def from_name(cls, colorname):
        if False:
            for i in range(10):
                print('nop')
        "Construct a BranchColor object by the color's name."
        return cls(*cls.color_names[colorname])

    def to_hex(self):
        if False:
            return 10
        "Return a 24-bit hexadecimal RGB representation of this color.\n\n        The returned string is suitable for use in HTML/CSS, as a color\n        parameter in matplotlib, and perhaps other situations.\n\n        Examples\n        --------\n        >>> bc = BranchColor(12, 200, 100)\n        >>> bc.to_hex()\n        '#0cc864'\n\n        "
        return f'#{self.red:02x}{self.green:02x}{self.blue:02x}'

    def to_rgb(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a tuple of RGB values (0 to 255) representing this color.\n\n        Examples\n        --------\n        >>> bc = BranchColor(255, 165, 0)\n        >>> bc.to_rgb()\n        (255, 165, 0)\n\n        '
        return (self.red, self.green, self.blue)

    def __repr__(self) -> str:
        if False:
            return 10
        'Preserve the standard RGB order when representing this object.'
        return '%s(red=%d, green=%d, blue=%d)' % (self.__class__.__name__, self.red, self.green, self.blue)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        "Show the color's RGB values."
        return '(%d, %d, %d)' % (self.red, self.green, self.blue)