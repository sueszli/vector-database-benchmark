"""Tests for open_spiel.python.environment.cliff_walking."""
import random
from absl.testing import absltest
from open_spiel.python import rl_environment
from open_spiel.python.environments import cliff_walking

def _select_random_legal_action(time_step):
    if False:
        i = 10
        return i + 15
    cur_legal_actions = time_step.observations['legal_actions'][0]
    action = random.choice(cur_legal_actions)
    return action

class CliffWalkingEnvTest(absltest.TestCase):

    def test_obs_spec(self):
        if False:
            for i in range(10):
                print('nop')
        env = cliff_walking.Environment()
        obs_specs = env.observation_spec()
        self.assertLen(obs_specs, 3)
        self.assertCountEqual(obs_specs.keys(), ['current_player', 'info_state', 'legal_actions'])
        self.assertEqual(obs_specs['info_state'], (2,))

    def test_action_spec(self):
        if False:
            i = 10
            return i + 15
        env = cliff_walking.Environment()
        action_spec = env.action_spec()
        self.assertLen(action_spec, 4)
        self.assertCountEqual(action_spec.keys(), ['dtype', 'max', 'min', 'num_actions'])
        self.assertEqual(action_spec['num_actions'], 4)
        self.assertEqual(action_spec['dtype'], int)

    def test_action_interfaces(self):
        if False:
            print('Hello World!')
        env = cliff_walking.Environment()
        time_step = env.reset()
        action_list = [cliff_walking.UP]
        time_step = env.step(action_list)
        self.assertEqual(time_step.step_type, rl_environment.StepType.MID)
        action_int = cliff_walking.UP
        time_step = env.step(action_int)
        self.assertEqual(time_step.step_type, rl_environment.StepType.MID)

    def test_many_runs(self):
        if False:
            while True:
                i = 10
        random.seed(1234)
        for _ in range(30):
            height = random.randint(3, 10)
            width = random.randint(3, 10)
            env = cliff_walking.Environment(height=height, width=width)
            time_step = env.reset()
            self.assertEqual(time_step.step_type, rl_environment.StepType.FIRST)
            self.assertIsNone(time_step.rewards)
            action_int = cliff_walking.UP
            time_step = env.step(action_int)
            self.assertEqual(time_step.step_type, rl_environment.StepType.MID)
            self.assertEqual(time_step.rewards, [-1.0])
            action_int = cliff_walking.RIGHT
            for _ in range(1, width):
                time_step = env.step(action_int)
                self.assertEqual(time_step.step_type, rl_environment.StepType.MID)
                self.assertEqual(time_step.rewards, [-1.0])
            action_int = cliff_walking.DOWN
            time_step = env.step(action_int)
            self.assertEqual(time_step.step_type, rl_environment.StepType.LAST)
            self.assertEqual(time_step.rewards, [-1.0])
if __name__ == '__main__':
    absltest.main()