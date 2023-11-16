"""Tests for topological sort."""
import pprint
from bzrlib.tests import TestCase
from bzrlib.tsort import topo_sort, TopoSorter, MergeSorter, merge_sort
from bzrlib.errors import GraphCycleError
from bzrlib.revision import NULL_REVISION

class TopoSortTests(TestCase):

    def assertSortAndIterate(self, graph, result_list):
        if False:
            print('Hello World!')
        'Check that sorting and iter_topo_order on graph works.'
        self.assertEqual(result_list, topo_sort(graph))
        self.assertEqual(result_list, list(TopoSorter(graph).iter_topo_order()))

    def assertSortAndIterateRaise(self, exception_type, graph):
        if False:
            return 10
        'Try both iterating and topo_sorting graph and expect an exception.'
        self.assertRaises(exception_type, topo_sort, graph)
        self.assertRaises(exception_type, list, TopoSorter(graph).iter_topo_order())

    def assertSortAndIterateOrder(self, graph):
        if False:
            print('Hello World!')
        "Check topo_sort and iter_topo_order is genuinely topological order.\n\n        For every child in the graph, check if it comes after all of it's\n        parents.\n        "
        sort_result = topo_sort(graph)
        iter_result = list(TopoSorter(graph).iter_topo_order())
        for (node, parents) in graph:
            for parent in parents:
                if sort_result.index(node) < sort_result.index(parent):
                    self.fail('parent %s must come before child %s:\n%s' % (parent, node, sort_result))
                if iter_result.index(node) < iter_result.index(parent):
                    self.fail('parent %s must come before child %s:\n%s' % (parent, node, iter_result))

    def test_tsort_empty(self):
        if False:
            i = 10
            return i + 15
        'TopoSort empty list'
        self.assertSortAndIterate([], [])

    def test_tsort_easy(self):
        if False:
            return 10
        'TopoSort list with one node'
        self.assertSortAndIterate({0: []}.items(), [0])

    def test_tsort_cycle(self):
        if False:
            return 10
        'TopoSort traps graph with cycles'
        self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [0]}.items())

    def test_tsort_cycle_2(self):
        if False:
            print('Hello World!')
        'TopoSort traps graph with longer cycle'
        self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [2], 2: [0]}.items())

    def test_topo_sort_cycle_with_tail(self):
        if False:
            return 10
        'TopoSort traps graph with longer cycle'
        self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [2], 2: [3, 4], 3: [0], 4: []}.items())

    def test_tsort_1(self):
        if False:
            i = 10
            return i + 15
        'TopoSort simple nontrivial graph'
        self.assertSortAndIterate({0: [3], 1: [4], 2: [1, 4], 3: [], 4: [0, 3]}.items(), [3, 0, 4, 1, 2])

    def test_tsort_partial(self):
        if False:
            return 10
        'Topological sort with partial ordering.\n\n        Multiple correct orderings are possible, so test for \n        correctness, not for exact match on the resulting list.\n        '
        self.assertSortAndIterateOrder([(0, []), (1, [0]), (2, [0]), (3, [0]), (4, [1, 2, 3]), (5, [1, 2]), (6, [1, 2]), (7, [2, 3]), (8, [0, 1, 4, 5, 6])])

    def test_tsort_unincluded_parent(self):
        if False:
            print('Hello World!')
        "Sort nodes, but don't include some parents in the output"
        self.assertSortAndIterate([(0, [1]), (1, [2])], [1, 0])

class MergeSortTests(TestCase):

    def assertSortAndIterate(self, graph, branch_tip, result_list, generate_revno, mainline_revisions=None):
        if False:
            while True:
                i = 10
        'Check that merge based sorting and iter_topo_order on graph works.'
        value = merge_sort(graph, branch_tip, mainline_revisions=mainline_revisions, generate_revno=generate_revno)
        if result_list != value:
            self.assertEqualDiff(pprint.pformat(result_list), pprint.pformat(value))
        self.assertEqual(result_list, list(MergeSorter(graph, branch_tip, mainline_revisions=mainline_revisions, generate_revno=generate_revno).iter_topo_order()))

    def test_merge_sort_empty(self):
        if False:
            i = 10
            return i + 15
        self.assertSortAndIterate({}, None, [], False)
        self.assertSortAndIterate({}, None, [], True)
        self.assertSortAndIterate({}, NULL_REVISION, [], False)
        self.assertSortAndIterate({}, NULL_REVISION, [], True)

    def test_merge_sort_not_empty_no_tip(self):
        if False:
            i = 10
            return i + 15
        self.assertSortAndIterate({0: []}.items(), None, [], False)
        self.assertSortAndIterate({0: []}.items(), None, [], True)

    def test_merge_sort_one_revision(self):
        if False:
            print('Hello World!')
        self.assertSortAndIterate({'id': []}.items(), 'id', [(0, 'id', 0, True)], False)
        self.assertSortAndIterate({'id': []}.items(), 'id', [(0, 'id', 0, (1,), True)], True)

    def test_sequence_numbers_increase_no_merges(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['B']}.items(), 'C', [(0, 'C', 0, False), (1, 'B', 0, False), (2, 'A', 0, True)], False)
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['B']}.items(), 'C', [(0, 'C', 0, (3,), False), (1, 'B', 0, (2,), False), (2, 'A', 0, (1,), True)], True)

    def test_sequence_numbers_increase_with_merges(self):
        if False:
            while True:
                i = 10
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A', 'B']}.items(), 'C', [(0, 'C', 0, False), (1, 'B', 1, True), (2, 'A', 0, True)], False)
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A', 'B']}.items(), 'C', [(0, 'C', 0, (2,), False), (1, 'B', 1, (1, 1, 1), True), (2, 'A', 0, (1,), True)], True)

    def test_merge_sort_race(self):
        if False:
            while True:
                i = 10
        graph = {'A': [], 'B': ['A'], 'C': ['B'], 'D': ['B', 'C'], 'F': ['B', 'D']}
        self.assertSortAndIterate(graph, 'F', [(0, 'F', 0, (3,), False), (1, 'D', 1, (2, 2, 1), False), (2, 'C', 2, (2, 1, 1), True), (3, 'B', 0, (2,), False), (4, 'A', 0, (1,), True)], True)
        graph = {'A': [], 'B': ['A'], 'C': ['B'], 'X': ['B'], 'D': ['X', 'C'], 'F': ['B', 'D']}
        self.assertSortAndIterate(graph, 'F', [(0, 'F', 0, (3,), False), (1, 'D', 1, (2, 1, 2), False), (2, 'C', 2, (2, 2, 1), True), (3, 'X', 1, (2, 1, 1), True), (4, 'B', 0, (2,), False), (5, 'A', 0, (1,), True)], True)

    def test_merge_depth_with_nested_merges(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertSortAndIterate({'A': ['D', 'B'], 'B': ['C', 'F'], 'C': ['H'], 'D': ['H', 'E'], 'E': ['G', 'F'], 'F': ['G'], 'G': ['H'], 'H': []}.items(), 'A', [(0, 'A', 0, False), (1, 'B', 1, False), (2, 'C', 1, True), (3, 'D', 0, False), (4, 'E', 1, False), (5, 'F', 2, True), (6, 'G', 1, True), (7, 'H', 0, True)], False)
        self.assertSortAndIterate({'A': ['D', 'B'], 'B': ['C', 'F'], 'C': ['H'], 'D': ['H', 'E'], 'E': ['G', 'F'], 'F': ['G'], 'G': ['H'], 'H': []}.items(), 'A', [(0, 'A', 0, (3,), False), (1, 'B', 1, (1, 3, 2), False), (2, 'C', 1, (1, 3, 1), True), (3, 'D', 0, (2,), False), (4, 'E', 1, (1, 1, 2), False), (5, 'F', 2, (1, 2, 1), True), (6, 'G', 1, (1, 1, 1), True), (7, 'H', 0, (1,), True)], True)

    def test_dotted_revnos_with_simple_merges(self):
        if False:
            print('Hello World!')
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A'], 'D': ['B'], 'E': ['C'], 'F': ['C'], 'G': ['D', 'E'], 'H': ['F'], 'I': ['F'], 'J': ['G', 'H'], 'K': ['I'], 'L': ['J', 'K']}.items(), 'L', [(0, 'L', 0, (6,), False), (1, 'K', 1, (1, 3, 2), False), (2, 'I', 1, (1, 3, 1), True), (3, 'J', 0, (5,), False), (4, 'H', 1, (1, 2, 2), False), (5, 'F', 1, (1, 2, 1), True), (6, 'G', 0, (4,), False), (7, 'E', 1, (1, 1, 2), False), (8, 'C', 1, (1, 1, 1), True), (9, 'D', 0, (3,), False), (10, 'B', 0, (2,), False), (11, 'A', 0, (1,), True)], True)
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A'], 'D': ['B'], 'E': ['C'], 'F': ['C'], 'G': ['D', 'E'], 'H': ['F'], 'I': ['F'], 'J': ['G', 'H'], 'K': ['I'], 'L': ['J', 'K'], 'M': ['A'], 'N': ['L', 'M']}.items(), 'N', [(0, 'N', 0, (7,), False), (1, 'M', 1, (1, 4, 1), True), (2, 'L', 0, (6,), False), (3, 'K', 1, (1, 3, 2), False), (4, 'I', 1, (1, 3, 1), True), (5, 'J', 0, (5,), False), (6, 'H', 1, (1, 2, 2), False), (7, 'F', 1, (1, 2, 1), True), (8, 'G', 0, (4,), False), (9, 'E', 1, (1, 1, 2), False), (10, 'C', 1, (1, 1, 1), True), (11, 'D', 0, (3,), False), (12, 'B', 0, (2,), False), (13, 'A', 0, (1,), True)], True)

    def test_end_of_merge_not_last_revision_in_branch(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertSortAndIterate({'A': ['B'], 'B': []}, 'A', [(0, 'A', 0, False), (1, 'B', 0, True)], False)
        self.assertSortAndIterate({'A': ['B'], 'B': []}, 'A', [(0, 'A', 0, (2,), False), (1, 'B', 0, (1,), True)], True)

    def test_end_of_merge_multiple_revisions_merged_at_once(self):
        if False:
            return 10
        self.assertSortAndIterate({'A': ['H', 'B', 'E'], 'B': ['D', 'C'], 'C': ['D'], 'D': ['H'], 'E': ['G', 'F'], 'F': ['G'], 'G': ['H'], 'H': []}, 'A', [(0, 'A', 0, False), (1, 'B', 1, False), (2, 'C', 2, True), (3, 'D', 1, True), (4, 'E', 1, False), (5, 'F', 2, True), (6, 'G', 1, True), (7, 'H', 0, True)], False)
        self.assertSortAndIterate({'A': ['H', 'B', 'E'], 'B': ['D', 'C'], 'C': ['D'], 'D': ['H'], 'E': ['G', 'F'], 'F': ['G'], 'G': ['H'], 'H': []}, 'A', [(0, 'A', 0, (2,), False), (1, 'B', 1, (1, 3, 2), False), (2, 'C', 2, (1, 4, 1), True), (3, 'D', 1, (1, 3, 1), True), (4, 'E', 1, (1, 1, 2), False), (5, 'F', 2, (1, 2, 1), True), (6, 'G', 1, (1, 1, 1), True), (7, 'H', 0, (1,), True)], True)

    def test_mainline_revs_partial(self):
        if False:
            while True:
                i = 10
        self.assertSortAndIterate({'A': ['E', 'B'], 'B': ['D', 'C'], 'C': ['D'], 'D': ['E'], 'E': []}, 'A', [(0, 'A', 0, False), (1, 'B', 0, False), (2, 'C', 1, True)], False, mainline_revisions=['D', 'B', 'A'])
        self.assertSortAndIterate({'A': ['E', 'B'], 'B': ['D', 'C'], 'C': ['D'], 'D': ['E'], 'E': []}, 'A', [(0, 'A', 0, (4,), False), (1, 'B', 0, (3,), False), (2, 'C', 1, (2, 1, 1), True)], True, mainline_revisions=['D', 'B', 'A'])

    def test_mainline_revs_with_none(self):
        if False:
            while True:
                i = 10
        self.assertSortAndIterate({'A': []}, 'A', [(0, 'A', 0, True)], False, mainline_revisions=[None, 'A'])
        self.assertSortAndIterate({'A': []}, 'A', [(0, 'A', 0, (1,), True)], True, mainline_revisions=[None, 'A'])

    def test_mainline_revs_with_ghost(self):
        if False:
            while True:
                i = 10
        self.assertSortAndIterate({'B': [], 'C': ['B']}.items(), 'C', [(0, 'C', 0, (2,), False), (1, 'B', 0, (1,), True)], True, mainline_revisions=['A', 'B', 'C'])

    def test_parallel_root_sequence_numbers_increase_with_merges(self):
        if False:
            for i in range(10):
                print('nop')
        'When there are parallel roots, check their revnos.'
        self.assertSortAndIterate({'A': [], 'B': [], 'C': ['A', 'B']}.items(), 'C', [(0, 'C', 0, (2,), False), (1, 'B', 1, (0, 1, 1), True), (2, 'A', 0, (1,), True)], True)

    def test_revnos_are_globally_assigned(self):
        if False:
            while True:
                i = 10
        'revnos are assigned according to the revision they derive from.'
        self.assertSortAndIterate({'J': ['G', 'I'], 'I': ['H'], 'H': ['A'], 'G': ['D', 'F'], 'F': ['E'], 'E': ['A'], 'D': ['A', 'C'], 'C': ['B'], 'B': ['A'], 'A': []}.items(), 'J', [(0, 'J', 0, (4,), False), (1, 'I', 1, (1, 3, 2), False), (2, 'H', 1, (1, 3, 1), True), (3, 'G', 0, (3,), False), (4, 'F', 1, (1, 2, 2), False), (5, 'E', 1, (1, 2, 1), True), (6, 'D', 0, (2,), False), (7, 'C', 1, (1, 1, 2), False), (8, 'B', 1, (1, 1, 1), True), (9, 'A', 0, (1,), True)], True)

    def test_roots_and_sub_branches_versus_ghosts(self):
        if False:
            while True:
                i = 10
        'Extra roots and their mini branches use the same numbering.\n\n        All of them use the 0-node numbering.\n        '
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['B'], 'D': [], 'E': ['D'], 'F': ['D'], 'G': ['E', 'F'], 'H': ['C', 'G'], 'I': [], 'J': ['H', 'I'], 'K': [], 'L': ['K'], 'M': ['K'], 'N': ['L', 'M'], 'O': ['N'], 'P': ['N'], 'Q': ['O', 'P'], 'R': ['J', 'Q']}.items(), 'R', [(0, 'R', 0, (6,), False), (1, 'Q', 1, (0, 4, 5), False), (2, 'P', 2, (0, 6, 1), True), (3, 'O', 1, (0, 4, 4), False), (4, 'N', 1, (0, 4, 3), False), (5, 'M', 2, (0, 5, 1), True), (6, 'L', 1, (0, 4, 2), False), (7, 'K', 1, (0, 4, 1), True), (8, 'J', 0, (5,), False), (9, 'I', 1, (0, 3, 1), True), (10, 'H', 0, (4,), False), (11, 'G', 1, (0, 1, 3), False), (12, 'F', 2, (0, 2, 1), True), (13, 'E', 1, (0, 1, 2), False), (14, 'D', 1, (0, 1, 1), True), (15, 'C', 0, (3,), False), (16, 'B', 0, (2,), False), (17, 'A', 0, (1,), True)], True)