"""Implementation of Graph algorithms when we have already loaded everything.
"""
from __future__ import absolute_import
from collections import deque
from bzrlib import errors, revision

class _KnownGraphNode(object):
    """Represents a single object in the known graph."""
    __slots__ = ('key', 'parent_keys', 'child_keys', 'gdfo')

    def __init__(self, key, parent_keys):
        if False:
            return 10
        self.key = key
        self.parent_keys = parent_keys
        self.child_keys = []
        self.gdfo = None

    def __repr__(self):
        if False:
            print('Hello World!')
        return '%s(%s  gdfo:%s par:%s child:%s)' % (self.__class__.__name__, self.key, self.gdfo, self.parent_keys, self.child_keys)

class _MergeSortNode(object):
    """Information about a specific node in the merge graph."""
    __slots__ = ('key', 'merge_depth', 'revno', 'end_of_merge')

    def __init__(self, key, merge_depth, revno, end_of_merge):
        if False:
            while True:
                i = 10
        self.key = key
        self.merge_depth = merge_depth
        self.revno = revno
        self.end_of_merge = end_of_merge

class KnownGraph(object):
    """This is a class which assumes we already know the full graph."""

    def __init__(self, parent_map, do_cache=True):
        if False:
            print('Hello World!')
        'Create a new KnownGraph instance.\n\n        :param parent_map: A dictionary mapping key => parent_keys\n        '
        self._nodes = {}
        self._known_heads = {}
        self.do_cache = do_cache
        self._initialize_nodes(parent_map)
        self._find_gdfo()

    def _initialize_nodes(self, parent_map):
        if False:
            print('Hello World!')
        'Populate self._nodes.\n\n        After this has finished:\n        - self._nodes will have an entry for every entry in parent_map.\n        - ghosts will have a parent_keys = None,\n        - all nodes found will also have .child_keys populated with all known\n          child_keys,\n        '
        nodes = self._nodes
        for (key, parent_keys) in parent_map.iteritems():
            if key in nodes:
                node = nodes[key]
                node.parent_keys = parent_keys
            else:
                node = _KnownGraphNode(key, parent_keys)
                nodes[key] = node
            for parent_key in parent_keys:
                try:
                    parent_node = nodes[parent_key]
                except KeyError:
                    parent_node = _KnownGraphNode(parent_key, None)
                    nodes[parent_key] = parent_node
                parent_node.child_keys.append(key)

    def _find_tails(self):
        if False:
            print('Hello World!')
        return [node for node in self._nodes.itervalues() if not node.parent_keys]

    def _find_tips(self):
        if False:
            i = 10
            return i + 15
        return [node for node in self._nodes.itervalues() if not node.child_keys]

    def _find_gdfo(self):
        if False:
            while True:
                i = 10
        nodes = self._nodes
        known_parent_gdfos = {}
        pending = []
        for node in self._find_tails():
            node.gdfo = 1
            pending.append(node)
        while pending:
            node = pending.pop()
            for child_key in node.child_keys:
                child = nodes[child_key]
                if child_key in known_parent_gdfos:
                    known_gdfo = known_parent_gdfos[child_key] + 1
                    present = True
                else:
                    known_gdfo = 1
                    present = False
                if child.gdfo is None or node.gdfo + 1 > child.gdfo:
                    child.gdfo = node.gdfo + 1
                if known_gdfo == len(child.parent_keys):
                    pending.append(child)
                    if present:
                        del known_parent_gdfos[child_key]
                else:
                    known_parent_gdfos[child_key] = known_gdfo

    def add_node(self, key, parent_keys):
        if False:
            return 10
        'Add a new node to the graph.\n\n        If this fills in a ghost, then the gdfos of all children will be\n        updated accordingly.\n        \n        :param key: The node being added. If this is a duplicate, this is a\n            no-op.\n        :param parent_keys: The parents of the given node.\n        :return: None (should we return if this was a ghost, etc?)\n        '
        nodes = self._nodes
        if key in nodes:
            node = nodes[key]
            if node.parent_keys is None:
                node.parent_keys = parent_keys
                self._known_heads.clear()
            else:
                parent_keys = list(parent_keys)
                existing_parent_keys = list(node.parent_keys)
                if parent_keys == existing_parent_keys:
                    return
                else:
                    raise ValueError('Parent key mismatch, existing node %s has parents of %s not %s' % (key, existing_parent_keys, parent_keys))
        else:
            node = _KnownGraphNode(key, parent_keys)
            nodes[key] = node
        parent_gdfo = 0
        for parent_key in parent_keys:
            try:
                parent_node = nodes[parent_key]
            except KeyError:
                parent_node = _KnownGraphNode(parent_key, None)
                parent_node.gdfo = 1
                nodes[parent_key] = parent_node
            if parent_gdfo < parent_node.gdfo:
                parent_gdfo = parent_node.gdfo
            parent_node.child_keys.append(key)
        node.gdfo = parent_gdfo + 1
        pending = deque([node])
        while pending:
            node = pending.popleft()
            next_gdfo = node.gdfo + 1
            for child_key in node.child_keys:
                child = nodes[child_key]
                if child.gdfo < next_gdfo:
                    child.gdfo = next_gdfo
                    pending.append(child)

    def heads(self, keys):
        if False:
            i = 10
            return i + 15
        'Return the heads from amongst keys.\n\n        This is done by searching the ancestries of each key.  Any key that is\n        reachable from another key is not returned; all the others are.\n\n        This operation scales with the relative depth between any two keys. It\n        uses gdfo to avoid walking all ancestry.\n\n        :param keys: An iterable of keys.\n        :return: A set of the heads. Note that as a set there is no ordering\n            information. Callers will need to filter their input to create\n            order if they need it.\n        '
        candidate_nodes = dict(((key, self._nodes[key]) for key in keys))
        if revision.NULL_REVISION in candidate_nodes:
            candidate_nodes.pop(revision.NULL_REVISION)
            if not candidate_nodes:
                return frozenset([revision.NULL_REVISION])
        if len(candidate_nodes) < 2:
            return frozenset(candidate_nodes)
        heads_key = frozenset(candidate_nodes)
        try:
            heads = self._known_heads[heads_key]
            return heads
        except KeyError:
            pass
        seen = set()
        pending = []
        min_gdfo = None
        for node in candidate_nodes.values():
            if node.parent_keys:
                pending.extend(node.parent_keys)
            if min_gdfo is None or node.gdfo < min_gdfo:
                min_gdfo = node.gdfo
        nodes = self._nodes
        while pending:
            node_key = pending.pop()
            if node_key in seen:
                continue
            seen.add(node_key)
            node = nodes[node_key]
            if node.gdfo <= min_gdfo:
                continue
            if node.parent_keys:
                pending.extend(node.parent_keys)
        heads = heads_key.difference(seen)
        if self.do_cache:
            self._known_heads[heads_key] = heads
        return heads

    def topo_sort(self):
        if False:
            print('Hello World!')
        'Return the nodes in topological order.\n\n        All parents must occur before all children.\n        '
        for node in self._nodes.itervalues():
            if node.gdfo is None:
                raise errors.GraphCycleError(self._nodes)
        pending = self._find_tails()
        pending_pop = pending.pop
        pending_append = pending.append
        topo_order = []
        topo_order_append = topo_order.append
        num_seen_parents = dict.fromkeys(self._nodes, 0)
        while pending:
            node = pending_pop()
            if node.parent_keys is not None:
                topo_order_append(node.key)
            for child_key in node.child_keys:
                child_node = self._nodes[child_key]
                seen_parents = num_seen_parents[child_key] + 1
                if seen_parents == len(child_node.parent_keys):
                    pending_append(child_node)
                    del num_seen_parents[child_key]
                else:
                    num_seen_parents[child_key] = seen_parents
        return topo_order

    def gc_sort(self):
        if False:
            for i in range(10):
                print('nop')
        "Return a reverse topological ordering which is 'stable'.\n\n        There are a few constraints:\n          1) Reverse topological (all children before all parents)\n          2) Grouped by prefix\n          3) 'stable' sorting, so that we get the same result, independent of\n             machine, or extra data.\n        To do this, we use the same basic algorithm as topo_sort, but when we\n        aren't sure what node to access next, we sort them lexicographically.\n        "
        tips = self._find_tips()
        prefix_tips = {}
        for node in tips:
            if node.key.__class__ is str or len(node.key) == 1:
                prefix = ''
            else:
                prefix = node.key[0]
            prefix_tips.setdefault(prefix, []).append(node)
        num_seen_children = dict.fromkeys(self._nodes, 0)
        result = []
        for prefix in sorted(prefix_tips):
            pending = sorted(prefix_tips[prefix], key=lambda n: n.key, reverse=True)
            while pending:
                node = pending.pop()
                if node.parent_keys is None:
                    continue
                result.append(node.key)
                for parent_key in sorted(node.parent_keys, reverse=True):
                    parent_node = self._nodes[parent_key]
                    seen_children = num_seen_children[parent_key] + 1
                    if seen_children == len(parent_node.child_keys):
                        pending.append(parent_node)
                        del num_seen_children[parent_key]
                    else:
                        num_seen_children[parent_key] = seen_children
        return result

    def merge_sort(self, tip_key):
        if False:
            for i in range(10):
                print('nop')
        'Compute the merge sorted graph output.'
        from bzrlib import tsort
        as_parent_map = dict(((node.key, node.parent_keys) for node in self._nodes.itervalues() if node.parent_keys is not None))
        return [_MergeSortNode(key, merge_depth, revno, end_of_merge) for (_, key, merge_depth, revno, end_of_merge) in tsort.merge_sort(as_parent_map, tip_key, mainline_revisions=None, generate_revno=True)]

    def get_parent_keys(self, key):
        if False:
            while True:
                i = 10
        'Get the parents for a key\n        \n        Returns a list containg the parents keys. If the key is a ghost,\n        None is returned. A KeyError will be raised if the key is not in\n        the graph.\n        \n        :param keys: Key to check (eg revision_id)\n        :return: A list of parents\n        '
        return self._nodes[key].parent_keys

    def get_child_keys(self, key):
        if False:
            i = 10
            return i + 15
        'Get the children for a key\n        \n        Returns a list containg the children keys. A KeyError will be raised\n        if the key is not in the graph.\n        \n        :param keys: Key to check (eg revision_id)\n        :return: A list of children\n        '
        return self._nodes[key].child_keys