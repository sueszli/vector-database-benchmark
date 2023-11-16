"""Implements directed graphs to sort and manipulate packages within a prefix.

Object inheritance:

.. autoapi-inheritance-diagram:: PrefixGraph GeneralGraph
   :top-classes: conda.models.prefix_graph.PrefixGraph
   :parts: 1
"""
from collections import defaultdict
from logging import getLogger
try:
    from boltons.setutils import IndexedSet
except ImportError:
    from .._vendor.boltons.setutils import IndexedSet
from ..base.context import context
from ..common.compat import on_win
from ..exceptions import CyclicalDependencyError
from .enums import NoarchType
from .match_spec import MatchSpec
log = getLogger(__name__)

class PrefixGraph:
    """
    A directed graph structure used for sorting packages (prefix_records) in prefixes and
    manipulating packages within prefixes (e.g. removing and pruning).

    The terminology used for edge direction is "parents" and "children" rather than "successors"
    and "predecessors". The parent nodes of a record are those records in the graph that
    match the record's "depends" field.  E.g. NodeA depends on NodeB, then NodeA is a child
    of NodeB, and NodeB is a parent of NodeA.  Nodes can have zero parents, or more than two
    parents.

    Most public methods mutate the graph.
    """

    def __init__(self, records, specs=()):
        if False:
            print('Hello World!')
        records = tuple(records)
        specs = set(specs)
        self.graph = graph = {}
        self.spec_matches = spec_matches = {}
        for node in records:
            parent_match_specs = tuple((MatchSpec(d) for d in node.depends))
            parent_nodes = {rec for rec in records if any((m.match(rec) for m in parent_match_specs))}
            graph[node] = parent_nodes
            matching_specs = IndexedSet((s for s in specs if s.match(node)))
            if matching_specs:
                spec_matches[node] = matching_specs
        self._toposort()

    def remove_spec(self, spec):
        if False:
            return 10
        '\n        Remove all matching nodes, and any associated child nodes.\n\n        Args:\n            spec (MatchSpec):\n\n        Returns:\n            tuple[PrefixRecord]: The removed nodes.\n\n        '
        node_matches = {node for node in self.graph if spec.match(node)}
        for feature_name in spec.get_raw_value('track_features') or ():
            feature_spec = MatchSpec(features=feature_name)
            node_matches.update((node for node in self.graph if feature_spec.match(node)))
        remove_these = set()
        for node in node_matches:
            remove_these.add(node)
            remove_these.update(self.all_descendants(node))
        remove_these = tuple(filter(lambda node: node in remove_these, self.graph))
        for node in remove_these:
            self._remove_node(node)
        self._toposort()
        return tuple(remove_these)

    def remove_youngest_descendant_nodes_with_specs(self):
        if False:
            i = 10
            return i + 15
        '\n        A specialized method used to determine only dependencies of requested specs.\n\n        Returns:\n            tuple[PrefixRecord]: The removed nodes.\n\n        '
        graph = self.graph
        spec_matches = self.spec_matches
        inverted_graph = {node: {key for key in graph if node in graph[key]} for node in graph}
        youngest_nodes_with_specs = tuple((node for (node, children) in inverted_graph.items() if not children and node in spec_matches))
        removed_nodes = tuple(filter(lambda node: node in youngest_nodes_with_specs, self.graph))
        for node in removed_nodes:
            self._remove_node(node)
        self._toposort()
        return removed_nodes

    @property
    def records(self):
        if False:
            i = 10
            return i + 15
        return iter(self.graph)

    def prune(self):
        if False:
            print('Hello World!')
        'Prune back all packages until all child nodes are anchored by a spec.\n\n        Returns:\n            tuple[PrefixRecord]: The pruned nodes.\n\n        '
        graph = self.graph
        spec_matches = self.spec_matches
        original_order = tuple(self.graph)
        removed_nodes = set()
        while True:
            inverted_graph = {node: {key for key in graph if node in graph[key]} for node in graph}
            prunable_nodes = tuple((node for (node, children) in inverted_graph.items() if not children and node not in spec_matches))
            if not prunable_nodes:
                break
            for node in prunable_nodes:
                removed_nodes.add(node)
                self._remove_node(node)
        removed_nodes = tuple(filter(lambda node: node in removed_nodes, original_order))
        self._toposort()
        return removed_nodes

    def get_node_by_name(self, name):
        if False:
            return 10
        return next((rec for rec in self.graph if rec.name == name))

    def all_descendants(self, node):
        if False:
            for i in range(10):
                print('nop')
        graph = self.graph
        inverted_graph = {node: {key for key in graph if node in graph[key]} for node in graph}
        nodes = [node]
        nodes_seen = set()
        q = 0
        while q < len(nodes):
            for child_node in inverted_graph[nodes[q]]:
                if child_node not in nodes_seen:
                    nodes_seen.add(child_node)
                    nodes.append(child_node)
            q += 1
        return tuple(filter(lambda node: node in nodes_seen, graph))

    def all_ancestors(self, node):
        if False:
            for i in range(10):
                print('nop')
        graph = self.graph
        nodes = [node]
        nodes_seen = set()
        q = 0
        while q < len(nodes):
            for parent_node in graph[nodes[q]]:
                if parent_node not in nodes_seen:
                    nodes_seen.add(parent_node)
                    nodes.append(parent_node)
            q += 1
        return tuple(filter(lambda node: node in nodes_seen, graph))

    def _remove_node(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Removes this node and all edges referencing it.'
        graph = self.graph
        if node not in graph:
            raise KeyError('node %s does not exist' % node)
        graph.pop(node)
        self.spec_matches.pop(node, None)
        for (node, edges) in graph.items():
            if node in edges:
                edges.remove(node)

    def _toposort(self):
        if False:
            for i in range(10):
                print('nop')
        graph_copy = {node: IndexedSet(parents) for (node, parents) in self.graph.items()}
        self._toposort_prepare_graph(graph_copy)
        if context.allow_cycles:
            sorted_nodes = tuple(self._topo_sort_handle_cycles(graph_copy))
        else:
            sorted_nodes = tuple(self._toposort_raise_on_cycles(graph_copy))
        original_graph = self.graph
        self.graph = {node: original_graph[node] for node in sorted_nodes}
        return sorted_nodes

    @classmethod
    def _toposort_raise_on_cycles(cls, graph):
        if False:
            for i in range(10):
                print('nop')
        if not graph:
            return
        while True:
            no_parent_nodes = IndexedSet(sorted((node for (node, parents) in graph.items() if len(parents) == 0), key=lambda x: x.name))
            if not no_parent_nodes:
                break
            for node in no_parent_nodes:
                yield node
                graph.pop(node, None)
            for parents in graph.values():
                parents -= no_parent_nodes
        if len(graph) != 0:
            raise CyclicalDependencyError(tuple(graph))

    @classmethod
    def _topo_sort_handle_cycles(cls, graph):
        if False:
            while True:
                i = 10
        for (k, v) in graph.items():
            v.discard(k)
        nodes_that_are_parents = {node for parents in graph.values() for node in parents}
        nodes_without_parents = (node for node in graph if not graph[node])
        disconnected_nodes = sorted((node for node in nodes_without_parents if node not in nodes_that_are_parents), key=lambda x: x.name)
        yield from disconnected_nodes
        t = cls._toposort_raise_on_cycles(graph)
        while True:
            try:
                value = next(t)
                yield value
            except CyclicalDependencyError as e:
                log.debug('%r', e)
                yield cls._toposort_pop_key(graph)
                t = cls._toposort_raise_on_cycles(graph)
                continue
            except StopIteration:
                return

    @staticmethod
    def _toposort_pop_key(graph):
        if False:
            i = 10
            return i + 15
        '\n        Pop an item from the graph that has the fewest parents.\n        In the case of a tie, use the node with the alphabetically-first package name.\n        '
        node_with_fewest_parents = sorted(((len(parents), node.dist_str(), node) for (node, parents) in graph.items()))[0][2]
        graph.pop(node_with_fewest_parents)
        for parents in graph.values():
            parents.discard(node_with_fewest_parents)
        return node_with_fewest_parents

    @staticmethod
    def _toposort_prepare_graph(graph):
        if False:
            return 10
        for node in graph:
            if node.name == 'python':
                parents = graph[node]
                for parent in tuple(parents):
                    if parent.name == 'pip':
                        parents.remove(parent)
        if on_win:
            menuinst_node = next((node for node in graph if node.name == 'menuinst'), None)
            python_node = next((node for node in graph if node.name == 'python'), None)
            if menuinst_node:
                assert python_node is not None
                menuinst_parents = graph[menuinst_node]
                for (node, parents) in graph.items():
                    if python_node in parents and node not in menuinst_parents:
                        parents.add(menuinst_node)
            conda_node = next((node for node in graph if node.name == 'conda'), None)
            if conda_node:
                conda_parents = graph[conda_node]
                for (node, parents) in graph.items():
                    if hasattr(node, 'noarch') and node.noarch == NoarchType.python and (node not in conda_parents):
                        parents.add(conda_node)

class GeneralGraph(PrefixGraph):
    """
    Compared with PrefixGraph, this class takes in more than one record of a given name,
    and operates on that graph from the higher view across any matching dependencies.  It is
    not a Prefix thing, but more like a "graph of all possible candidates" thing, and is used
    for unsatisfiability analysis
    """

    def __init__(self, records, specs=()):
        if False:
            print('Hello World!')
        records = tuple(records)
        super().__init__(records, specs)
        self.specs_by_name = defaultdict(dict)
        for node in records:
            parent_dict = self.specs_by_name.get(node.name, {})
            for dep in tuple((MatchSpec(d) for d in node.depends)):
                deps = parent_dict.get(dep.name, set())
                deps.add(dep)
                parent_dict[dep.name] = deps
            self.specs_by_name[node.name] = parent_dict
        consolidated_graph = {}
        for (node, parent_nodes) in reversed(list(self.graph.items())):
            cg = consolidated_graph.get(node.name, set())
            cg.update((_.name for _ in parent_nodes))
            consolidated_graph[node.name] = cg
        self.graph_by_name = consolidated_graph

    def breadth_first_search_by_name(self, root_spec, target_spec):
        if False:
            print('Hello World!')
        'Return shorted path from root_spec to spec_name'
        queue = []
        queue.append([root_spec])
        visited = []
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node in visited:
                continue
            visited.append(node)
            if node == target_spec:
                return path
            children = []
            specs = self.specs_by_name.get(node.name)
            if specs is None:
                continue
            for (_, deps) in specs.items():
                children.extend(list(deps))
            for adj in children:
                if adj.name == target_spec.name and adj.version != target_spec.version:
                    pass
                else:
                    new_path = list(path)
                    new_path.append(adj)
                    queue.append(new_path)