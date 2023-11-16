from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType
(tf1, tf, tfv) = try_import_tf()
(torch, _) = try_import_torch()

@PublicAPI
class PolynomialSchedule(Schedule):
    """Polynomial interpolation between `initial_p` and `final_p`.

    Over `schedule_timesteps`. After this many time steps, always returns
    `final_p`.
    """

    def __init__(self, schedule_timesteps: int, final_p: float, framework: Optional[str], initial_p: float=1.0, power: float=2.0):
        if False:
            while True:
                i = 10
        'Initializes a PolynomialSchedule instance.\n\n        Args:\n            schedule_timesteps: Number of time steps for which to\n                linearly anneal initial_p to final_p\n            final_p: Final output value.\n            framework: The framework descriptor string, e.g. "tf",\n                "torch", or None.\n            initial_p: Initial output value.\n            power: The exponent to use (default: quadratic).\n        '
        super().__init__(framework=framework)
        assert schedule_timesteps > 0
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.power = power

    @override(Schedule)
    def _value(self, t: TensorType) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        'Returns the result of:\n        final_p + (initial_p - final_p) * (1 - `t`/t_max) ** power\n        '
        if self.framework == 'torch' and torch and isinstance(t, torch.Tensor):
            t = t.float()
        t = min(t, self.schedule_timesteps)
        return self.final_p + (self.initial_p - self.final_p) * (1.0 - t / self.schedule_timesteps) ** self.power

    @override(Schedule)
    def _tf_value_op(self, t: TensorType) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        t = tf.math.minimum(t, self.schedule_timesteps)
        return self.final_p + (self.initial_p - self.final_p) * (1.0 - t / self.schedule_timesteps) ** self.power