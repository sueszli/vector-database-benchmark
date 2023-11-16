"""Helper functions for TPU device names."""
from typing import Text
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['tpu.core'])
def core(num: int) -> Text:
    if False:
        return 10
    'Returns the device name for a core in a replicated TPU computation.\n\n  Args:\n    num: the virtual core number within each replica to which operators should\n    be assigned.\n  Returns:\n    A device name, suitable for passing to `tf.device()`.\n  '
    return 'device:TPU_REPLICATED_CORE:{}'.format(num)