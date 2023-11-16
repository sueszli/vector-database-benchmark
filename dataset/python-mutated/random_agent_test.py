"""Tests for open_spiel.python.algorithms.random_agent."""
from absl.testing import absltest
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

class RandomAgentTest(absltest.TestCase):

    def test_step(self):
        if False:
            while True:
                i = 10
        agent = random_agent.RandomAgent(player_id=0, num_actions=10)
        legal_actions = [0, 2, 3, 5]
        time_step = rl_environment.TimeStep(observations={'info_state': [[0], [1]], 'legal_actions': [legal_actions, []], 'current_player': 0}, rewards=None, discounts=None, step_type=None)
        agent_output = agent.step(time_step)
        self.assertIn(agent_output.action, legal_actions)
        self.assertAlmostEqual(sum(agent_output.probs), 1.0)
        self.assertEqual(len([x for x in agent_output.probs if x > 0]), len(legal_actions))
        self.assertTrue(np.allclose(agent_output.probs[legal_actions], [0.25] * 4, atol=1e-05))
if __name__ == '__main__':
    absltest.main()