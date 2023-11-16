import gymnasium as gym
from typing import Type

class ActionTransform(gym.ActionWrapper):

    def __init__(self, env, low, high):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(env)
        self._low = low
        self._high = high
        self.action_space = type(env.action_space)(self._low, self._high, env.action_space.shape, env.action_space.dtype)

    def action(self, action):
        if False:
            print('Hello World!')
        return (action - self._low) / (self._high - self._low) * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low

def transform_action_space(env_name_or_creator) -> Type[gym.Env]:
    if False:
        print('Hello World!')
    'Wrapper for gym.Envs to have their action space transformed.\n\n    Args:\n        env_name_or_creator (Union[str, Callable[]]: String specifier or\n            env_maker function.\n\n    Returns:\n        New transformed_action_space_env function that returns an environment\n        wrapped by the ActionTransform wrapper. The constructor takes a\n        config dict with `_low` and `_high` keys specifying the new action\n        range (default -1.0 to 1.0). The reset of the config dict will be\n        passed on to the underlying/wrapped env\'s constructor.\n\n    .. testcode::\n        :skipif: True\n\n        # By gym string:\n        pendulum_300_to_500_cls = transform_action_space("Pendulum-v1")\n        # Create a transformed pendulum env.\n        pendulum_300_to_500 = pendulum_300_to_500_cls({"_low": -15.0})\n        pendulum_300_to_500.action_space\n\n    .. testoutput::\n\n        gym.spaces.Box(-15.0, 1.0, (1, ), "float32")\n    '

    def transformed_action_space_env(config):
        if False:
            return 10
        if isinstance(env_name_or_creator, str):
            inner_env = gym.make(env_name_or_creator)
        else:
            inner_env = env_name_or_creator(config)
        _low = config.pop('low', -1.0)
        _high = config.pop('high', 1.0)
        env = ActionTransform(inner_env, _low, _high)
        return env
    return transformed_action_space_env
TransformedActionPendulum = transform_action_space('Pendulum-v1')