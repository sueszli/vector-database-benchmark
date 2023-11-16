import pickle
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python import test_utils
import pyspiel
SPIEL_SAMPLED_STOCHASTIC_GAMES_LIST = [g for g in pyspiel.registered_games() if g.default_loadable and g.chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC]
assert len(SPIEL_SAMPLED_STOCHASTIC_GAMES_LIST) >= 2
NUM_RUNS = 2

class SampledStochasticGamesTest(parameterized.TestCase):

    @parameterized.parameters(*SPIEL_SAMPLED_STOCHASTIC_GAMES_LIST)
    def test_stateful_game_serialization(self, game_info):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game(game_info.short_name, {'rng_seed': 0})
        for seed in range(NUM_RUNS):
            test_utils.random_playout(game.new_initial_state(), seed)
            deserialized_game = pickle.loads(pickle.dumps(game))
            state = test_utils.random_playout(game.new_initial_state(), seed)
            deserialized_state = test_utils.random_playout(deserialized_game.new_initial_state(), seed)
            self.assertEqual(str(state), str(deserialized_state))
if __name__ == '__main__':
    absltest.main()