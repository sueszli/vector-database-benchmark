from abc import ABCMeta, abstractmethod
from gymnasium.spaces import Discrete
import numpy as np
from pathlib import Path
import unittest
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.test_utils import check, framework_iterator
(tf1, tf, tfv) = try_import_tf()
(torch, _) = try_import_torch()

class DummyComponent:
    """A simple class that can be used for testing framework-agnostic logic.

    Implements a simple `add()` method for adding a value to
    `self.prop_b`.
    """

    def __init__(self, prop_a, prop_b=0.5, prop_c=None, framework='tf', **kwargs):
        if False:
            return 10
        self.framework = framework
        self.prop_a = prop_a
        self.prop_b = prop_b
        self.prop_c = prop_c or 'default'
        self.prop_d = kwargs.pop('prop_d', 4)
        self.kwargs = kwargs

    def add(self, value):
        if False:
            print('Hello World!')
        if self.framework == 'tf':
            return self._add_tf(value)
        return self.prop_b + value

    def _add_tf(self, value):
        if False:
            return 10
        return tf.add(self.prop_b, value)

class NonAbstractChildOfDummyComponent(DummyComponent):
    pass

class AbstractDummyComponent(DummyComponent, metaclass=ABCMeta):
    """Used for testing `from_config()`."""

    @abstractmethod
    def some_abstract_method(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class TestFrameWorkAgnosticComponents(unittest.TestCase):
    """
    Tests the Component base class to implement framework-agnostic functional
    units.
    """

    def test_dummy_components(self):
        if False:
            return 10
        script_dir = Path(__file__).parent
        abs_path = script_dir.absolute()
        for (fw, sess) in framework_iterator(session=True):
            test = from_config({'type': AbstractDummyComponent, 'framework': fw})
            check(test, None)
            component = from_config(dict(type=DummyComponent, prop_a=1.0, prop_d='non_default', framework=fw))
            check(component.prop_d, 'non_default')
            config_file = str(abs_path.joinpath('dummy_config.json'))
            component = from_config(config_file, framework=fw)
            check(component.prop_c, 'default')
            check(component.prop_d, 4)
            value = component.add(3.3)
            if sess:
                value = sess.run(value)
            check(value, 5.3)
            config_file = str(abs_path.joinpath('dummy_config.yml'))
            component = from_config(config_file, framework=fw)
            check(component.prop_a, 'something else')
            check(component.prop_d, 3)
            value = component.add(1.2)
            if sess:
                value = sess.run(value)
            check(value, np.array([2.2]))
            component = from_config('{"type": "ray.rllib.utils.tests.test_framework_agnostic_components.DummyComponent", "prop_a": "A", "prop_b": -1.0, "prop_c": "non-default", "framework": "' + fw + '"}')
            check(component.prop_a, 'A')
            check(component.prop_d, 4)
            value = component.add(-1.1)
            if sess:
                value = sess.run(value)
            check(value, -2.1)
            component = from_config(DummyComponent, '{"type": "NonAbstractChildOfDummyComponent", "prop_a": "A", "prop_b": -1.0, "prop_c": "non-default","framework": "' + fw + '"}')
            check(component.prop_a, 'A')
            check(component.prop_d, 4)
            value = component.add(-1.1)
            if sess:
                value = sess.run(value)
            check(value, -2.1)
            scope = None
            if sess:
                scope = tf1.variable_scope('exploration_object')
                scope.__enter__()
            component = from_config(Exploration, {'type': 'EpsilonGreedy', 'action_space': Discrete(2), 'framework': fw, 'num_workers': 0, 'worker_index': 0, 'policy_config': {}, 'model': None})
            if scope:
                scope.__exit__(None, None, None)
            check(component.epsilon_schedule.outside_value, 0.05)
            component = from_config('type: ray.rllib.utils.tests.test_framework_agnostic_components.DummyComponent\nprop_a: B\nprop_b: -1.5\nprop_c: non-default\nframework: {}'.format(fw))
            check(component.prop_a, 'B')
            check(component.prop_d, 4)
            value = component.add(-5.1)
            if sess:
                value = sess.run(value)
            check(value, np.array([-6.6]))

    def test_unregistered_envs(self):
        if False:
            i = 10
            return i + 15
        'Tests, whether an Env can be specified simply by its absolute class.'
        env_cls = 'ray.rllib.examples.env.stateless_cartpole.StatelessCartPole'
        env = from_config(env_cls, {'config': 42.0})
        (state, _) = env.reset()
        self.assertTrue(state.shape == (2,))
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))