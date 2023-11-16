"""Tests for pyspiel Chat Game."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.games import chat_game
from open_spiel.python.games.chat_games.configs import config_fixed_mock
from open_spiel.python.games.chat_games.configs import config_rnd_mock
from open_spiel.python.games.chat_games.utils import test_utils as chat_test_utils
import pyspiel
GLOBAL_TEST_LLM = chat_test_utils.TestLLM.MOCK

class ChatGameTest(parameterized.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.fixed_config = config_fixed_mock.get_config()
        self.random_config = config_rnd_mock.get_config()
        vectorizer = chat_test_utils.MockVectorizer()
        self.vectorize = vectorizer.vectorize

    @parameterized.named_parameters(dict(testcase_name='fixed_scenario', fixed_scenario=True), dict(testcase_name='random_scenario', fixed_scenario=False))
    def test_game_from_cc(self, fixed_scenario):
        if False:
            i = 10
            return i + 15
        'Runs our standard game tests, checking API consistency.'
        if fixed_scenario:
            config = self.fixed_config
        else:
            config = self.random_config
        game = pyspiel.load_game('chat_game', config.params.to_dict())
        game.load_chat_game(llm_type=GLOBAL_TEST_LLM, vectorize=self.vectorize, seed=1234, **config.game)
        pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)
if __name__ == '__main__':
    absltest.main()