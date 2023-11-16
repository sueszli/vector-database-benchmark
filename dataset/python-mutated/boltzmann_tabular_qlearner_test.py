"""Tests for open_spiel.python.algorithms.boltzmann_tabular_qlearner."""
from absl.testing import absltest
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import boltzmann_tabular_qlearner
import pyspiel
SEED = 10000
SIMPLE_EFG_DATA = '\n  EFG 2 R "Simple single-agent problem" { "Player 1" } ""\n  p "ROOT" 1 1 "ROOT" { "L" "R" } 0\n    t "L" 1 "Outcome L" { -1.0 }\n    t "R" 2 "Outcome R" { 1.0 }\n'

class BoltzmannQlearnerTest(absltest.TestCase):

    def test_simple_game(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)
        env = rl_environment.Environment(game=game)
        agent = boltzmann_tabular_qlearner.BoltzmannQLearner(0, game.num_distinct_actions())
        total_reward = 0
        for _ in range(100):
            total_eval_reward = 0
            for _ in range(1000):
                time_step = env.reset()
                while not time_step.last():
                    agent_output = agent.step(time_step)
                    time_step = env.step([agent_output.action])
                    total_reward += time_step.rewards[0]
                agent.step(time_step)
            self.assertGreaterEqual(total_reward, 75)
            for _ in range(1000):
                time_step = env.reset()
                while not time_step.last():
                    agent_output = agent.step(time_step, is_evaluation=True)
                    time_step = env.step([agent_output.action])
                    total_eval_reward += time_step.rewards[0]
            self.assertGreaterEqual(total_eval_reward, 250)
if __name__ == '__main__':
    np.random.seed(SEED)
    absltest.main()