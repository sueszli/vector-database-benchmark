"""Tests for open_spiel.python.algorithms.playthrough."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import generate_playthrough

class PlaythroughTest(absltest.TestCase):

    def test_runs(self):
        if False:
            for i in range(10):
                print('nop')
        result = generate_playthrough.playthrough('tic_tac_toe', action_sequence=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertNotEmpty(result)

    def test_format_tensor_1d(self):
        if False:
            while True:
                i = 10
        lines = generate_playthrough._format_tensor(np.array((1, 0, 1, 1)), 'x')
        self.assertEqual(lines, ['x: ◉◯◉◉'])

    def test_format_tensor_2d(self):
        if False:
            return 10
        lines = generate_playthrough._format_tensor(np.array(((1, 0), (1, 1))), 'x')
        self.assertEqual(lines, ['x: ◉◯', '   ◉◉'])

    def test_format_tensor_3d(self):
        if False:
            for i in range(10):
                print('nop')
        lines = []
        tensor = np.array((((1, 0), (1, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 0))))
        lines = generate_playthrough._format_tensor(tensor, 'x')
        self.assertEqual(lines, ['x:', '◉◯  ◯◯  ◯◉', '◉◉  ◉◯  ◉◯'])

    def test_format_tensor_3d_linewrap(self):
        if False:
            while True:
                i = 10
        tensor = np.array((((1, 0), (1, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 0))))
        lines = generate_playthrough._format_tensor(tensor, 'x', max_cols=9)
        self.assertEqual(lines, ['x:', '◉◯  ◯◯', '◉◉  ◉◯', '', '◯◉', '◉◯'])
if __name__ == '__main__':
    absltest.main()