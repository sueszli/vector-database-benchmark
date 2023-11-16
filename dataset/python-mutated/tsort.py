"""Topological sorting routines."""
from __future__ import absolute_import
from bzrlib import errors, graph as _mod_graph, revision as _mod_revision
__all__ = ['topo_sort', 'TopoSorter', 'merge_sort', 'MergeSorter']

def topo_sort(graph):
    if False:
        i = 10
        return i + 15
    'Topological sort a graph.\n\n    graph -- sequence of pairs of node->parents_list.\n\n    The result is a list of node names, such that all parents come before their\n    children.\n\n    node identifiers can be any hashable object, and are typically strings.\n\n    This function has the same purpose as the TopoSorter class, but uses a\n    different algorithm to sort the graph. That means that while both return a\n    list with parents before their child nodes, the exact ordering can be\n    different.\n\n    topo_sort is faster when the whole list is needed, while when iterating\n    over a part of the list, TopoSorter.iter_topo_order should be used.\n    '
    kg = _mod_graph.KnownGraph(dict(graph))
    return kg.topo_sort()

class TopoSorter(object):

    def __init__(self, graph):
        if False:
            print('Hello World!')
        "Topological sorting of a graph.\n\n        :param graph: sequence of pairs of node_name->parent_names_list.\n                      i.e. [('C', ['B']), ('B', ['A']), ('A', [])]\n                      For this input the output from the sort or\n                      iter_topo_order routines will be:\n                      'A', 'B', 'C'\n\n        node identifiers can be any hashable object, and are typically strings.\n\n        If you have a graph like [('a', ['b']), ('a', ['c'])] this will only use\n        one of the two values for 'a'.\n\n        The graph is sorted lazily: until you iterate or sort the input is\n        not processed other than to create an internal representation.\n\n        iteration or sorting may raise GraphCycleError if a cycle is present\n        in the graph.\n        "
        self._graph = dict(graph)

    def sorted(self):
        if False:
            for i in range(10):
                print('nop')
        'Sort the graph and return as a list.\n\n        After calling this the sorter is empty and you must create a new one.\n        '
        return list(self.iter_topo_order())

    def iter_topo_order(self):
        if False:
            return 10
        'Yield the nodes of the graph in a topological order.\n\n        After finishing iteration the sorter is empty and you cannot continue\n        iteration.\n        '
        graph = self._graph
        visitable = set(graph)
        pending_node_stack = []
        pending_parents_stack = []
        completed_node_names = set()
        while graph:
            (node_name, parents) = graph.popitem()
            pending_node_stack.append(node_name)
            pending_parents_stack.append(list(parents))
            while pending_node_stack:
                parents_to_visit = pending_parents_stack[-1]
                if not parents_to_visit:
                    popped_node = pending_node_stack.pop()
                    pending_parents_stack.pop()
                    completed_node_names.add(popped_node)
                    yield popped_node
                else:
                    next_node_name = parents_to_visit.pop()
                    if next_node_name in completed_node_names:
                        continue
                    if next_node_name not in visitable:
                        continue
                    try:
                        parents = graph.pop(next_node_name)
                    except KeyError:
                        raise errors.GraphCycleError(pending_node_stack)
                    pending_node_stack.append(next_node_name)
                    pending_parents_stack.append(list(parents))

def merge_sort(graph, branch_tip, mainline_revisions=None, generate_revno=False):
    if False:
        i = 10
        return i + 15
    'Topological sort a graph which groups merges.\n\n    :param graph: sequence of pairs of node->parents_list.\n    :param branch_tip: the tip of the branch to graph. Revisions not\n                       reachable from branch_tip are not included in the\n                       output.\n    :param mainline_revisions: If not None this forces a mainline to be\n                               used rather than synthesised from the graph.\n                               This must be a valid path through some part\n                               of the graph. If the mainline does not cover all\n                               the revisions, output stops at the start of the\n                               old revision listed in the mainline revisions\n                               list.\n                               The order for this parameter is oldest-first.\n    :param generate_revno: Optional parameter controlling the generation of\n        revision number sequences in the output. See the output description of\n        the MergeSorter docstring for details.\n    :result: See the MergeSorter docstring for details.\n\n    Node identifiers can be any hashable object, and are typically strings.\n    '
    return MergeSorter(graph, branch_tip, mainline_revisions, generate_revno).sorted()

class MergeSorter(object):
    __slots__ = ['_node_name_stack', '_node_merge_depth_stack', '_pending_parents_stack', '_first_child_stack', '_left_subtree_pushed_stack', '_generate_revno', '_graph', '_mainline_revisions', '_stop_revision', '_original_graph', '_revnos', '_revno_to_branch_count', '_completed_node_names', '_scheduled_nodes']

    def __init__(self, graph, branch_tip, mainline_revisions=None, generate_revno=False):
        if False:
            while True:
                i = 10
        "Merge-aware topological sorting of a graph.\n\n        :param graph: sequence of pairs of node_name->parent_names_list.\n                      i.e. [('C', ['B']), ('B', ['A']), ('A', [])]\n                      For this input the output from the sort or\n                      iter_topo_order routines will be:\n                      'A', 'B', 'C'\n        :param branch_tip: the tip of the branch to graph. Revisions not\n                       reachable from branch_tip are not included in the\n                       output.\n        :param mainline_revisions: If not None this forces a mainline to be\n                               used rather than synthesised from the graph.\n                               This must be a valid path through some part\n                               of the graph. If the mainline does not cover all\n                               the revisions, output stops at the start of the\n                               old revision listed in the mainline revisions\n                               list.\n                               The order for this parameter is oldest-first.\n        :param generate_revno: Optional parameter controlling the generation of\n            revision number sequences in the output. See the output description\n            for more details.\n\n        The result is a list sorted so that all parents come before\n        their children. Each element of the list is a tuple containing:\n        (sequence_number, node_name, merge_depth, end_of_merge)\n         * sequence_number: The sequence of this row in the output. Useful for\n           GUIs.\n         * node_name: The node name: opaque text to the merge routine.\n         * merge_depth: How many levels of merging deep this node has been\n           found.\n         * revno_sequence: When requested this field provides a sequence of\n             revision numbers for all revisions. The format is:\n             (REVNO, BRANCHNUM, BRANCHREVNO). BRANCHNUM is the number of the\n             branch that the revno is on. From left to right the REVNO numbers\n             are the sequence numbers within that branch of the revision.\n             For instance, the graph {A:[], B:['A'], C:['A', 'B']} will get\n             the following revno_sequences assigned: A:(1,), B:(1,1,1), C:(2,).\n             This should be read as 'A is the first commit in the trunk',\n             'B is the first commit on the first branch made from A', 'C is the\n             second commit in the trunk'.\n         * end_of_merge: When True the next node is part of a different merge.\n\n\n        node identifiers can be any hashable object, and are typically strings.\n\n        If you have a graph like [('a', ['b']), ('a', ['c'])] this will only use\n        one of the two values for 'a'.\n\n        The graph is sorted lazily: until you iterate or sort the input is\n        not processed other than to create an internal representation.\n\n        iteration or sorting may raise GraphCycleError if a cycle is present\n        in the graph.\n\n        Background information on the design:\n        -------------------------------------\n        definition: the end of any cluster or 'merge' occurs when:\n            1 - the next revision has a lower merge depth than we do.\n              i.e.\n              A 0\n              B  1\n              C   2\n              D  1\n              E 0\n              C, D are the ends of clusters, E might be but we need more data.\n            2 - or the next revision at our merge depth is not our left most\n              ancestor.\n              This is required to handle multiple-merges in one commit.\n              i.e.\n              A 0    [F, B, E]\n              B  1   [D, C]\n              C   2  [D]\n              D  1   [F]\n              E  1   [F]\n              F 0\n              C is the end of a cluster due to rule 1.\n              D is not the end of a cluster from rule 1, but is from rule 2: E\n                is not its left most ancestor\n              E is the end of a cluster due to rule 1\n              F might be but we need more data.\n\n        we show connecting lines to a parent when:\n         - The parent is the start of a merge within this cluster.\n           That is, the merge was not done to the mainline before this cluster\n           was merged to the mainline.\n           This can be detected thus:\n            * The parent has a higher merge depth and is the next revision in\n              the list.\n\n          The next revision in the list constraint is needed for this case:\n          A 0   [D, B]\n          B  1  [C, F]   # we do not want to show a line to F which is depth 2\n                           but not a merge\n          C  1  [H]      # note that this is a long line to show back to the\n                           ancestor - see the end of merge rules.\n          D 0   [G, E]\n          E  1  [G, F]\n          F   2 [G]\n          G  1  [H]\n          H 0\n         - Part of this merges 'branch':\n          The parent has the same merge depth and is our left most parent and we\n           are not the end of the cluster.\n          A 0   [C, B] lines: [B, C]\n          B  1  [E, C] lines: [C]\n          C 0   [D]    lines: [D]\n          D 0   [F, E] lines: [E, F]\n          E  1  [F]    lines: [F]\n          F 0\n         - The end of this merge/cluster:\n          we can ONLY have multiple parents at the end of a cluster if this\n          branch was previously merged into the 'mainline'.\n          - if we have one and only one parent, show it\n            Note that this may be to a greater merge depth - for instance if\n            this branch continued from a deeply nested branch to add something\n            to it.\n          - if we have more than one parent - show the second oldest (older ==\n            further down the list) parent with\n            an equal or lower merge depth\n             XXXX revisit when awake. ddaa asks about the relevance of each one\n             - maybe more than one parent is relevant\n        "
        self._generate_revno = generate_revno
        self._graph = dict(graph)
        if mainline_revisions is None:
            self._mainline_revisions = []
            self._stop_revision = None
        else:
            self._mainline_revisions = list(mainline_revisions)
            self._stop_revision = self._mainline_revisions[0]
        for (index, revision) in enumerate(self._mainline_revisions[1:]):
            parent = self._mainline_revisions[index]
            if parent is None:
                continue
            graph_parent_ids = self._graph[revision]
            if not graph_parent_ids:
                continue
            if graph_parent_ids[0] == parent:
                continue
            self._graph[revision].remove(parent)
            self._graph[revision].insert(0, parent)
        self._original_graph = dict(self._graph.items())
        self._revnos = dict(((revision, [None, True]) for revision in self._graph))
        self._revno_to_branch_count = {}
        self._node_name_stack = []
        self._node_merge_depth_stack = []
        self._pending_parents_stack = []
        self._first_child_stack = []
        self._completed_node_names = set()
        self._scheduled_nodes = []
        self._left_subtree_pushed_stack = []
        if branch_tip is not None and branch_tip != _mod_revision.NULL_REVISION and (branch_tip != (_mod_revision.NULL_REVISION,)):
            parents = self._graph.pop(branch_tip)
            self._push_node(branch_tip, 0, parents)

    def sorted(self):
        if False:
            print('Hello World!')
        'Sort the graph and return as a list.\n\n        After calling this the sorter is empty and you must create a new one.\n        '
        return list(self.iter_topo_order())

    def iter_topo_order(self):
        if False:
            print('Hello World!')
        'Yield the nodes of the graph in a topological order.\n\n        After finishing iteration the sorter is empty and you cannot continue\n        iteration.\n        '
        node_name_stack = self._node_name_stack
        node_merge_depth_stack = self._node_merge_depth_stack
        pending_parents_stack = self._pending_parents_stack
        left_subtree_pushed_stack = self._left_subtree_pushed_stack
        completed_node_names = self._completed_node_names
        scheduled_nodes = self._scheduled_nodes
        graph_pop = self._graph.pop

        def push_node(node_name, merge_depth, parents, node_name_stack_append=node_name_stack.append, node_merge_depth_stack_append=node_merge_depth_stack.append, left_subtree_pushed_stack_append=left_subtree_pushed_stack.append, pending_parents_stack_append=pending_parents_stack.append, first_child_stack_append=self._first_child_stack.append, revnos=self._revnos):
            if False:
                return 10
            'Add node_name to the pending node stack.\n\n            Names in this stack will get emitted into the output as they are popped\n            off the stack.\n\n            This inlines a lot of self._variable.append functions as local\n            variables.\n            '
            node_name_stack_append(node_name)
            node_merge_depth_stack_append(merge_depth)
            left_subtree_pushed_stack_append(False)
            pending_parents_stack_append(list(parents))
            parent_info = None
            if parents:
                try:
                    parent_info = revnos[parents[0]]
                except KeyError:
                    pass
            if parent_info is not None:
                first_child = parent_info[1]
                parent_info[1] = False
            else:
                first_child = None
            first_child_stack_append(first_child)

        def pop_node(node_name_stack_pop=node_name_stack.pop, node_merge_depth_stack_pop=node_merge_depth_stack.pop, first_child_stack_pop=self._first_child_stack.pop, left_subtree_pushed_stack_pop=left_subtree_pushed_stack.pop, pending_parents_stack_pop=pending_parents_stack.pop, original_graph=self._original_graph, revnos=self._revnos, completed_node_names_add=self._completed_node_names.add, scheduled_nodes_append=scheduled_nodes.append, revno_to_branch_count=self._revno_to_branch_count):
            if False:
                i = 10
                return i + 15
            'Pop the top node off the stack\n\n            The node is appended to the sorted output.\n            '
            node_name = node_name_stack_pop()
            merge_depth = node_merge_depth_stack_pop()
            first_child = first_child_stack_pop()
            left_subtree_pushed_stack_pop()
            pending_parents_stack_pop()
            parents = original_graph[node_name]
            parent_revno = None
            if parents:
                try:
                    parent_revno = revnos[parents[0]][0]
                except KeyError:
                    pass
            if parent_revno is not None:
                if not first_child:
                    base_revno = parent_revno[0]
                    branch_count = revno_to_branch_count.get(base_revno, 0)
                    branch_count += 1
                    revno_to_branch_count[base_revno] = branch_count
                    revno = (parent_revno[0], branch_count, 1)
                else:
                    revno = parent_revno[:-1] + (parent_revno[-1] + 1,)
            else:
                root_count = revno_to_branch_count.get(0, -1)
                root_count += 1
                if root_count:
                    revno = (0, root_count, 1)
                else:
                    revno = (1,)
                revno_to_branch_count[0] = root_count
            revnos[node_name][0] = revno
            completed_node_names_add(node_name)
            scheduled_nodes_append((node_name, merge_depth, revno))
            return node_name
        while node_name_stack:
            parents_to_visit = pending_parents_stack[-1]
            if not parents_to_visit:
                pop_node()
            else:
                while pending_parents_stack[-1]:
                    if not left_subtree_pushed_stack[-1]:
                        next_node_name = pending_parents_stack[-1].pop(0)
                        is_left_subtree = True
                        left_subtree_pushed_stack[-1] = True
                    else:
                        next_node_name = pending_parents_stack[-1].pop()
                        is_left_subtree = False
                    if next_node_name in completed_node_names:
                        continue
                    try:
                        parents = graph_pop(next_node_name)
                    except KeyError:
                        if next_node_name in self._original_graph:
                            raise errors.GraphCycleError(node_name_stack)
                        else:
                            continue
                    next_merge_depth = 0
                    if is_left_subtree:
                        next_merge_depth = 0
                    else:
                        next_merge_depth = 1
                    next_merge_depth = node_merge_depth_stack[-1] + next_merge_depth
                    push_node(next_node_name, next_merge_depth, parents)
                    break
        sequence_number = 0
        stop_revision = self._stop_revision
        generate_revno = self._generate_revno
        original_graph = self._original_graph
        while scheduled_nodes:
            (node_name, merge_depth, revno) = scheduled_nodes.pop()
            if node_name == stop_revision:
                return
            if not len(scheduled_nodes):
                end_of_merge = True
            elif scheduled_nodes[-1][1] < merge_depth:
                end_of_merge = True
            elif scheduled_nodes[-1][1] == merge_depth and scheduled_nodes[-1][0] not in original_graph[node_name]:
                end_of_merge = True
            else:
                end_of_merge = False
            if generate_revno:
                yield (sequence_number, node_name, merge_depth, revno, end_of_merge)
            else:
                yield (sequence_number, node_name, merge_depth, end_of_merge)
            sequence_number += 1

    def _push_node(self, node_name, merge_depth, parents):
        if False:
            while True:
                i = 10
        'Add node_name to the pending node stack.\n\n        Names in this stack will get emitted into the output as they are popped\n        off the stack.\n        '
        self._node_name_stack.append(node_name)
        self._node_merge_depth_stack.append(merge_depth)
        self._left_subtree_pushed_stack.append(False)
        self._pending_parents_stack.append(list(parents))
        parent_info = None
        if parents:
            try:
                parent_info = self._revnos[parents[0]]
            except KeyError:
                pass
        if parent_info is not None:
            first_child = parent_info[1]
            parent_info[1] = False
        else:
            first_child = None
        self._first_child_stack.append(first_child)

    def _pop_node(self):
        if False:
            return 10
        'Pop the top node off the stack\n\n        The node is appended to the sorted output.\n        '
        node_name = self._node_name_stack.pop()
        merge_depth = self._node_merge_depth_stack.pop()
        first_child = self._first_child_stack.pop()
        self._left_subtree_pushed_stack.pop()
        self._pending_parents_stack.pop()
        parents = self._original_graph[node_name]
        parent_revno = None
        if parents:
            try:
                parent_revno = self._revnos[parents[0]][0]
            except KeyError:
                pass
        if parent_revno is not None:
            if not first_child:
                base_revno = parent_revno[0]
                branch_count = self._revno_to_branch_count.get(base_revno, 0)
                branch_count += 1
                self._revno_to_branch_count[base_revno] = branch_count
                revno = (parent_revno[0], branch_count, 1)
            else:
                revno = parent_revno[:-1] + (parent_revno[-1] + 1,)
        else:
            root_count = self._revno_to_branch_count.get(0, 0)
            root_count = self._revno_to_branch_count.get(0, -1)
            root_count += 1
            if root_count:
                revno = (0, root_count, 1)
            else:
                revno = (1,)
            self._revno_to_branch_count[0] = root_count
        self._revnos[node_name][0] = revno
        self._completed_node_names.add(node_name)
        self._scheduled_nodes.append((node_name, merge_depth, self._revnos[node_name][0]))
        return node_name