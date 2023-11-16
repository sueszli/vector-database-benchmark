"""Example of a custom observation filter

This example shows:
  - using a custom observation filter

"""
import argparse
import numpy as np
import ray
from ray import air, tune
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import get_trainable_cls
(tf1, tf, tfv) = try_import_tf()
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='PPO', help='The RLlib-registered algorithm to use.')
parser.add_argument('--stop-iters', type=int, default=200)

class SimpleRollingStat:

    def __init__(self, n=0, m=0, s=0):
        if False:
            for i in range(10):
                print('nop')
        self._n = n
        self._m = m
        self._s = s

    def copy(self):
        if False:
            return 10
        return SimpleRollingStat(self._n, self._m, self._s)

    def push(self, x):
        if False:
            for i in range(10):
                print('nop')
        self._n += 1
        delta = x - self._m
        self._m += delta / self._n
        self._s += delta * delta * (self._n - 1) / self._n

    def update(self, other):
        if False:
            print('Hello World!')
        n1 = self._n
        n2 = other.num_pushes
        n = n1 + n2
        if n == 0:
            return
        delta = self._m - other._m
        delta2 = delta * delta
        self._n = n
        self._m = (n1 * self._m + n2 * other._m) / n
        self._s = self._s + other._s + delta2 * n1 * n2 / n

    @property
    def n(self):
        if False:
            i = 10
            return i + 15
        return self._n

    @property
    def mean(self):
        if False:
            print('Hello World!')
        return self._m

    @property
    def var(self):
        if False:
            while True:
                i = 10
        return self._s / (self._n - 1) if self._n > 1 else np.square(self._m)

    @property
    def std(self):
        if False:
            while True:
                i = 10
        return np.sqrt(self.var)

class CustomFilter(Filter):
    """
    Filter that normalizes by using a single mean
    and std sampled from all obs inputs
    """
    is_concurrent = False

    def __init__(self, shape):
        if False:
            print('Hello World!')
        self.rs = SimpleRollingStat()
        self.buffer = SimpleRollingStat()
        self.shape = shape

    def reset_buffer(self) -> None:
        if False:
            print('Hello World!')
        self.buffer = SimpleRollingStat(self.shape)

    def apply_changes(self, other, with_buffer=False):
        if False:
            while True:
                i = 10
        self.rs.update(other.buffer)
        if with_buffer:
            self.buffer = other.buffer.copy()

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        other = CustomFilter(self.shape)
        other.sync(self)
        return other

    def as_serializable(self):
        if False:
            i = 10
            return i + 15
        return self.copy()

    def sync(self, other):
        if False:
            print('Hello World!')
        assert other.shape == self.shape, "Shapes don't match!"
        self.rs = other.rs.copy()
        self.buffer = other.buffer.copy()

    def __call__(self, x, update=True):
        if False:
            print('Hello World!')
        x = np.asarray(x)
        if update:
            if len(x.shape) == len(self.shape) + 1:
                for i in range(x.shape[0]):
                    self.push_stats(x[i], (self.rs, self.buffer))
            else:
                self.push_stats(x, (self.rs, self.buffer))
        x = x - self.rs.mean
        x = x / (self.rs.std + 1e-08)
        return x

    @staticmethod
    def push_stats(vector, buffers):
        if False:
            return 10
        for x in vector:
            for buffer in buffers:
                buffer.push(x)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'CustomFilter({self.shape}, {self.rs}, {self.buffer})'
if __name__ == '__main__':
    args = parser.parse_args()
    ray.init()
    config = get_trainable_cls(args.run).get_default_config().environment('CartPole-v1').rollouts(num_rollout_workers=0, observation_filter=lambda size: CustomFilter(size))
    tuner = tune.Tuner(args.run, param_space=config.to_dict(), run_config=air.RunConfig(stop={'training_iteration': args.stop_iters}))
    tuner.fit()
    ray.shutdown()