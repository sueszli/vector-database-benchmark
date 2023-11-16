"""Unit test for uniform random bot."""
import random
from absl.testing import absltest
from open_spiel.python.bots import uniform_random
import pyspiel

class BotTest(absltest.TestCase):

    def test_policy_is_uniform(self):
        if False:
            return 10
        game = pyspiel.load_game('leduc_poker')
        bots = [uniform_random.UniformRandomBot(0, random), uniform_random.UniformRandomBot(1, random)]
        state = game.new_initial_state()
        state.apply_action(2)
        state.apply_action(4)
        (policy, _) = bots[0].step_with_policy(state)
        self.assertCountEqual(policy, [(1, 0.5), (2, 0.5)])
        state.apply_action(2)
        (policy, _) = bots[1].step_with_policy(state)
        self.assertCountEqual(policy, [(0, 1 / 3), (1, 1 / 3), (2, 1 / 3)])

    def test_no_legal_actions(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('kuhn_poker')
        bot = uniform_random.UniformRandomBot(0, random)
        state = game.new_initial_state()
        state.apply_action(2)
        state.apply_action(1)
        state.apply_action(1)
        state.apply_action(0)
        bot.restart_at(state)
        (policy, action) = bot.step_with_policy(state)
        self.assertEqual(policy, [])
        self.assertEqual(action, pyspiel.INVALID_ACTION)
if __name__ == '__main__':
    absltest.main()