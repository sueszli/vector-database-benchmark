from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType
(tf1, tf, tfv) = try_import_tf()

@PublicAPI
class ConstantSchedule(Schedule):
    """A Schedule where the value remains constant over time."""

    def __init__(self, value: float, framework: Optional[str]=None):
        if False:
            while True:
                i = 10
        'Initializes a ConstantSchedule instance.\n\n        Args:\n            value: The constant value to return, independently of time.\n            framework: The framework descriptor string, e.g. "tf",\n                "torch", or None.\n        '
        super().__init__(framework=framework)
        self._v = value

    @override(Schedule)
    def _value(self, t: TensorType) -> TensorType:
        if False:
            i = 10
            return i + 15
        return self._v

    @override(Schedule)
    def _tf_value_op(self, t: TensorType) -> TensorType:
        if False:
            return 10
        return tf.constant(self._v)