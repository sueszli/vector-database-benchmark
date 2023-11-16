from algorithms.bfs import count_islands, maze_search, ladder_length
import unittest

class TestCountIslands(unittest.TestCase):

    def test_count_islands(self):
        if False:
            i = 10
            return i + 15
        grid_1 = [[1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
        self.assertEqual(1, count_islands(grid_1))
        grid_2 = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]
        self.assertEqual(3, count_islands(grid_2))
        grid_3 = [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0]]
        self.assertEqual(3, count_islands(grid_3))
        grid_4 = [[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 0, 0]]
        self.assertEqual(5, count_islands(grid_4))

class TestMazeSearch(unittest.TestCase):

    def test_maze_search(self):
        if False:
            for i in range(10):
                print('nop')
        grid_1 = [[1, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1], [1, 1, 1, 0, 1, 1]]
        self.assertEqual(14, maze_search(grid_1))
        grid_2 = [[1, 0, 0], [0, 1, 1], [0, 1, 1]]
        self.assertEqual(-1, maze_search(grid_2))

class TestWordLadder(unittest.TestCase):

    def test_ladder_length(self):
        if False:
            return 10
        self.assertEqual(5, ladder_length('hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log']))
        self.assertEqual(5, ladder_length('pick', 'tank', ['tock', 'tick', 'sank', 'sink', 'sick']))
        self.assertEqual(1, ladder_length('live', 'life', ['hoho', 'luck']))
        self.assertEqual(0, ladder_length('ate', 'ate', []))
        self.assertEqual(-1, ladder_length('rahul', 'coder', ['blahh', 'blhah']))
if __name__ == '__main__':
    unittest.main()