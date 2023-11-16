"""A vectorized RL Environment."""

class SyncVectorEnv(object):
    """A vectorized RL Environment.

  This environment is synchronized - games do not execute in parallel. Speedups
  are realized by calling models on many game states simultaneously.
  """

    def __init__(self, envs):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(envs, list):
            raise ValueError('Need to call this with a list of rl_environment.Environment objects')
        self.envs = envs

    def __len__(self):
        if False:
            return 10
        return len(self.envs)

    def observation_spec(self):
        if False:
            i = 10
            return i + 15
        return self.envs[0].observation_spec()

    @property
    def num_players(self):
        if False:
            for i in range(10):
                print('nop')
        return self.envs[0].num_players

    def step(self, step_outputs, reset_if_done=False):
        if False:
            return 10
        'Apply one step.\n\n    Args:\n      step_outputs: the step outputs\n      reset_if_done: if True, automatically reset the environment\n          when the epsiode ends\n\n    Returns:\n      time_steps: the time steps,\n      reward: the reward\n      done: done flag\n      unreset_time_steps: unreset time steps\n    '
        time_steps = [self.envs[i].step([step_outputs[i].action]) for i in range(len(self.envs))]
        reward = [step.rewards for step in time_steps]
        done = [step.last() for step in time_steps]
        unreset_time_steps = time_steps
        if reset_if_done:
            time_steps = self.reset(envs_to_reset=done)
        return (time_steps, reward, done, unreset_time_steps)

    def reset(self, envs_to_reset=None):
        if False:
            return 10
        if envs_to_reset is None:
            envs_to_reset = [True for _ in range(len(self.envs))]
        time_steps = [self.envs[i].reset() if envs_to_reset[i] else self.envs[i].get_time_step() for i in range(len(self.envs))]
        return time_steps