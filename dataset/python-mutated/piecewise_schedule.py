from typing import Callable, List, Optional, Tuple
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType
(tf1, tf, tfv) = try_import_tf()

def _linear_interpolation(left, right, alpha):
    if False:
        i = 10
        return i + 15
    return left + alpha * (right - left)

@PublicAPI
class PiecewiseSchedule(Schedule):
    """Implements a Piecewise Scheduler."""

    def __init__(self, endpoints: List[Tuple[int, float]], framework: Optional[str]=None, interpolation: Callable[[TensorType, TensorType, TensorType], TensorType]=_linear_interpolation, outside_value: Optional[float]=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a PiecewiseSchedule instance.\n\n        Args:\n            endpoints: A list of tuples\n                `(t, value)` such that the output\n                is an interpolation (given by the `interpolation` callable)\n                between two values.\n                E.g.\n                t=400 and endpoints=[(0, 20.0),(500, 30.0)]\n                output=20.0 + 0.8 * (30.0 - 20.0) = 28.0\n                NOTE: All the values for time must be sorted in an increasing\n                order.\n            framework: The framework descriptor string, e.g. "tf",\n                "torch", or None.\n            interpolation: A function that takes the left-value,\n                the right-value and an alpha interpolation parameter\n                (0.0=only left value, 1.0=only right value), which is the\n                fraction of distance from left endpoint to right endpoint.\n            outside_value: If t in call to `value` is\n                outside of all the intervals in `endpoints` this value is\n                returned. If None then an AssertionError is raised when outside\n                value is requested.\n        '
        super().__init__(framework=framework)
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self.interpolation = interpolation
        self.outside_value = outside_value
        self.endpoints = [(int(e[0]), float(e[1])) for e in endpoints]

    @override(Schedule)
    def _value(self, t: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        for ((l_t, l), (r_t, r)) in zip(self.endpoints[:-1], self.endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self.interpolation(l, r, alpha)
        assert self.outside_value is not None
        return self.outside_value

    @override(Schedule)
    def _tf_value_op(self, t: TensorType) -> TensorType:
        if False:
            return 10
        assert self.outside_value is not None, 'tf-version of PiecewiseSchedule requires `outside_value` to be provided!'
        endpoints = tf.cast(tf.stack([e[0] for e in self.endpoints] + [-1]), tf.int64)
        results_list = []
        for ((l_t, l), (r_t, r)) in zip(self.endpoints[:-1], self.endpoints[1:]):
            alpha = tf.cast(t - l_t, tf.float32) / tf.cast(r_t - l_t, tf.float32)
            results_list.append(self.interpolation(l, r, alpha))
        results_list.append(self.outside_value)
        results_list = tf.stack(results_list)

        def _cond(i, x):
            if False:
                for i in range(10):
                    print('nop')
            x = tf.cast(x, tf.int64)
            return tf.logical_not(tf.logical_or(tf.equal(endpoints[i + 1], -1), tf.logical_and(endpoints[i] <= x, x < endpoints[i + 1])))

        def _body(i, x):
            if False:
                i = 10
                return i + 15
            return (i + 1, t)
        idx_and_t = tf.while_loop(_cond, _body, [tf.constant(0, dtype=tf.int64), t])
        return results_list[idx_and_t[0]]