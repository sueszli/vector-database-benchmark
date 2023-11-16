"""Tests for open_spiel.python.algorithms.mcts."""
import math
import random
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.algorithms import mcts
import pyspiel
UCT_C = math.sqrt(2)

def _get_action(state, action_str):
    if False:
        while True:
            i = 10
    for action in state.legal_actions():
        if action_str == state.action_to_string(state.current_player(), action):
            return action
    raise ValueError('invalid action string: {}'.format(action_str))

def search_tic_tac_toe_state(initial_actions):
    if False:
        for i in range(10):
            print('nop')
    game = pyspiel.load_game('tic_tac_toe')
    state = game.new_initial_state()
    for action_str in initial_actions.split(' '):
        state.apply_action(_get_action(state, action_str))
    rng = np.random.RandomState(42)
    bot = mcts.MCTSBot(game, UCT_C, max_simulations=10000, solve=True, random_state=rng, evaluator=mcts.RandomRolloutEvaluator(n_rollouts=20, random_state=rng))
    return (bot.mcts_search(state), state)

def make_node(action, player=0, prior=1, **kwargs):
    if False:
        return 10
    node = mcts.SearchNode(action, player, prior)
    for (k, v) in kwargs.items():
        setattr(node, k, v)
    return node

class MctsBotTest(absltest.TestCase):

    def assertTTTStateStr(self, state, expected):
        if False:
            i = 10
            return i + 15
        expected = expected.replace(' ', '').strip()
        self.assertEqual(str(state), expected)

    def test_can_play_tic_tac_toe(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('tic_tac_toe')
        max_simulations = 100
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=20)
        bots = [mcts.MCTSBot(game, UCT_C, max_simulations, evaluator), mcts.MCTSBot(game, UCT_C, max_simulations, evaluator)]
        v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
        self.assertEqual(v[0] + v[1], 0)

    def test_can_play_both_sides(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('tic_tac_toe')
        bot = mcts.MCTSBot(game, UCT_C, max_simulations=100, evaluator=mcts.RandomRolloutEvaluator(n_rollouts=20))
        bots = [bot, bot]
        v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
        self.assertEqual(v[0] + v[1], 0)

    def test_can_play_single_player(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('catch')
        max_simulations = 100
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=20)
        bots = [mcts.MCTSBot(game, UCT_C, max_simulations, evaluator)]
        v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
        self.assertGreater(v[0], 0)

    def test_throws_on_simultaneous_game(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('matrix_mp')
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=20)
        with self.assertRaises(ValueError):
            mcts.MCTSBot(game, UCT_C, max_simulations=100, evaluator=evaluator)

    def test_can_play_three_player_stochastic_games(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('pig(players=3,winscore=20,horizon=30)')
        max_simulations = 100
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=5)
        bots = [mcts.MCTSBot(game, UCT_C, max_simulations, evaluator), mcts.MCTSBot(game, UCT_C, max_simulations, evaluator), mcts.MCTSBot(game, UCT_C, max_simulations, evaluator)]
        v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
        self.assertEqual(sum(v), 0)

    def test_solve_draw(self):
        if False:
            for i in range(10):
                print('nop')
        (root, state) = search_tic_tac_toe_state('x(1,1) o(0,0) x(2,2)')
        self.assertTTTStateStr(state, '\n        o..\n        .x.\n        ..x\n    ')
        self.assertEqual(root.outcome[root.player], 0)
        for c in root.children:
            self.assertLessEqual(c.outcome[c.player], 0)
        best = root.best_child()
        self.assertEqual(best.outcome[best.player], 0)
        self.assertIn(state.action_to_string(best.player, best.action), ('o(0,2)', 'o(2,0)'))

    def test_solve_loss(self):
        if False:
            return 10
        (root, state) = search_tic_tac_toe_state('x(1,1) o(0,0) x(2,2) o(0,1) x(0,2)')
        self.assertTTTStateStr(state, '\n        oox\n        .x.\n        ..x\n    ')
        self.assertEqual(root.outcome[root.player], -1)
        for c in root.children:
            self.assertEqual(c.outcome[c.player], -1)

    def test_solve_win(self):
        if False:
            print('Hello World!')
        (root, state) = search_tic_tac_toe_state('x(0,1) o(2,2)')
        self.assertTTTStateStr(state, '\n        .x.\n        ...\n        ..o\n    ')
        self.assertEqual(root.outcome[root.player], 1)
        best = root.best_child()
        self.assertEqual(best.outcome[best.player], 1)
        self.assertEqual(state.action_to_string(best.player, best.action), 'x(0,2)')

    def assertBestChild(self, choice, children):
        if False:
            print('Hello World!')
        random.shuffle(children)
        root = make_node(-1, children=children)
        self.assertEqual(root.best_child().action, choice)

    def test_choose_most_visited_when_not_solved(self):
        if False:
            while True:
                i = 10
        self.assertBestChild(0, [make_node(0, explore_count=50, total_reward=30), make_node(1, explore_count=40, total_reward=40)])

    def test_choose_win_over_most_visited(self):
        if False:
            return 10
        self.assertBestChild(1, [make_node(0, explore_count=50, total_reward=30), make_node(1, explore_count=40, total_reward=40, outcome=[1])])

    def test_choose_best_over_good(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertBestChild(1, [make_node(0, explore_count=50, total_reward=30, outcome=[0.5]), make_node(1, explore_count=40, total_reward=40, outcome=[0.8])])

    def test_choose_bad_over_worst(self):
        if False:
            while True:
                i = 10
        self.assertBestChild(0, [make_node(0, explore_count=50, total_reward=30, outcome=[-0.5]), make_node(1, explore_count=40, total_reward=40, outcome=[-0.8])])

    def test_choose_positive_reward_over_promising(self):
        if False:
            return 10
        self.assertBestChild(1, [make_node(0, explore_count=50, total_reward=40), make_node(1, explore_count=10, total_reward=1, outcome=[0.1])])

    def test_choose_most_visited_over_loss(self):
        if False:
            i = 10
            return i + 15
        self.assertBestChild(0, [make_node(0, explore_count=50, total_reward=30), make_node(1, explore_count=40, total_reward=40, outcome=[-1])])

    def test_choose_most_visited_over_draw(self):
        if False:
            i = 10
            return i + 15
        self.assertBestChild(0, [make_node(0, explore_count=50, total_reward=30), make_node(1, explore_count=40, total_reward=40, outcome=[0])])

    def test_choose_uncertainty_over_most_visited_loss(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertBestChild(1, [make_node(0, explore_count=50, total_reward=30, outcome=[-1]), make_node(1, explore_count=40, total_reward=40)])

    def test_choose_slowest_loss(self):
        if False:
            while True:
                i = 10
        self.assertBestChild(1, [make_node(0, explore_count=50, total_reward=10, outcome=[-1]), make_node(1, explore_count=60, total_reward=15, outcome=[-1])])
if __name__ == '__main__':
    absltest.main()