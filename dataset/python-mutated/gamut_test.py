"""Unit test for GamutGenerator."""
from absl import app
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel

class GamutGeneratorTest(parameterized.TestCase):

    def _gamut_generator(self):
        if False:
            print('Hello World!')
        return pyspiel.GamutGenerator('gamut.jar')

    @parameterized.parameters('-g BertrandOligopoly -players 2 -actions 4 -random_params', '-g UniformLEG-CG -players 2 -actions 4 -random_params', '-g PolymatrixGame-SW -players 2 -actions 4 -random_params', '-g GraphicalGame-SW -players 2 -actions 4 -random_params', '-g BidirectionalLEG-CG -players 2 -actions 4 -random_params', '-g CovariantGame -players 2 -actions 4 -random_params', '-g DispersionGame -players 2 -actions 4 -random_params', '-g MinimumEffortGame -players 2 -actions 4 -random_params', '-g RandomGame -players 2 -actions 4 -random_params', '-g TravelersDilemma -players 2 -actions 4 -random_params')
    def test_generate_game(self, game_str):
        if False:
            print('Hello World!')
        generator = self._gamut_generator()
        game = generator.generate_game(game_str)
        self.assertIsNotNone(game)
        payoff_tensor = game_payoffs_array(game)
        self.assertEqual(payoff_tensor.shape, (2, 4, 4))

    def test_gamut_api(self):
        if False:
            for i in range(10):
                print('nop')
        generator = self._gamut_generator()
        game = generator.generate_game('-g RandomGame -players 4 -normalize -min_payoff 0 ' + '-max_payoff 150 -actions 2 4 5 7')
        self.assertIsNotNone(game)
        game = generator.generate_game(['-g', 'RandomGame', '-players', '4', '-normalize', '-min_payoff', '0', '-max_payoff', '150', '-actions', '2', '4', '5', '7'])
        self.assertIsNotNone(game)
        matrix_game = generator.generate_matrix_game(['-g', 'RandomGame', '-players', '2', '-normalize', '-min_payoff', '0', '-max_payoff', '150', '-actions', '10', '15'])
        self.assertIsNotNone(matrix_game)
        print(matrix_game.new_initial_state())
        payoff_matrix = game_payoffs_array(matrix_game)
        print(payoff_matrix.shape)
        print(payoff_matrix)
        tensor_game = generator.generate_game(['-g', 'RandomGame', '-players', '4', '-normalize', '-min_payoff', '0', '-max_payoff', '150', '-actions', '2', '4', '5', '7'])
        self.assertIsNotNone(tensor_game)
        payoff_tensor = game_payoffs_array(tensor_game)
        print(payoff_tensor.shape)

def main(_):
    if False:
        print('Hello World!')
    absltest.main()
if __name__ == '__main__':
    app.run(main)