from typing import Dict, Iterable, List, Sequence
from synapse.util.iterutils import chunk_seq, sorted_topologically
from tests.unittest import TestCase

class ChunkSeqTests(TestCase):

    def test_short_seq(self) -> None:
        if False:
            return 10
        parts = chunk_seq('123', 8)
        self.assertEqual(list(parts), ['123'])

    def test_long_seq(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        parts = chunk_seq('abcdefghijklmnop', 8)
        self.assertEqual(list(parts), ['abcdefgh', 'ijklmnop'])

    def test_uneven_parts(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        parts = chunk_seq('abcdefghijklmnop', 5)
        self.assertEqual(list(parts), ['abcde', 'fghij', 'klmno', 'p'])

    def test_empty_input(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        parts: Iterable[Sequence] = chunk_seq([], 5)
        self.assertEqual(list(parts), [])

class SortTopologically(TestCase):

    def test_empty(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that an empty graph works correctly'
        graph: Dict[int, List[int]] = {}
        self.assertEqual(list(sorted_topologically([], graph)), [])

    def test_handle_empty_graph(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Test that a graph where a node doesn't have an entry is treated as empty"
        graph: Dict[int, List[int]] = {}
        self.assertEqual(list(sorted_topologically([1, 2], graph)), [1, 2])

    def test_disconnected(self) -> None:
        if False:
            return 10
        'Test that a graph with no edges work'
        graph: Dict[int, List[int]] = {1: [], 2: []}
        self.assertEqual(list(sorted_topologically([1, 2], graph)), [1, 2])

    def test_linear(self) -> None:
        if False:
            print('Hello World!')
        'Test that a simple `4 -> 3 -> 2 -> 1` graph works'
        graph: Dict[int, List[int]] = {1: [], 2: [1], 3: [2], 4: [3]}
        self.assertEqual(list(sorted_topologically([4, 3, 2, 1], graph)), [1, 2, 3, 4])

    def test_subset(self) -> None:
        if False:
            return 10
        'Test that only sorting a subset of the graph works'
        graph: Dict[int, List[int]] = {1: [], 2: [1], 3: [2], 4: [3]}
        self.assertEqual(list(sorted_topologically([4, 3], graph)), [3, 4])

    def test_fork(self) -> None:
        if False:
            while True:
                i = 10
        'Test that a forked graph works'
        graph: Dict[int, List[int]] = {1: [], 2: [1], 3: [1], 4: [2, 3]}
        self.assertEqual(list(sorted_topologically([4, 3, 2, 1], graph)), [1, 2, 3, 4])

    def test_duplicates(self) -> None:
        if False:
            while True:
                i = 10
        'Test that a graph with duplicate edges work'
        graph: Dict[int, List[int]] = {1: [], 2: [1, 1], 3: [2, 2], 4: [3]}
        self.assertEqual(list(sorted_topologically([4, 3, 2, 1], graph)), [1, 2, 3, 4])

    def test_multiple_paths(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that a graph with multiple paths between two nodes work'
        graph: Dict[int, List[int]] = {1: [], 2: [1], 3: [2], 4: [3, 2, 1]}
        self.assertEqual(list(sorted_topologically([4, 3, 2, 1], graph)), [1, 2, 3, 4])