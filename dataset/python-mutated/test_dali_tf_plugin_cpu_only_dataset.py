import nose_utils
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
from test_utils_tensorflow import skip_for_incompatible_tf
import tensorflow as tf
import random
import numpy as np

@pipeline_def()
def get_dali_pipe(value):
    if False:
        i = 10
        return i + 15
    data = types.Constant(value)
    return data

def test_dali_tf_dataset_cpu_only():
    if False:
        i = 10
        return i + 15
    skip_for_incompatible_tf()
    try:
        tf.compat.v1.enable_eager_execution()
    except Exception:
        pass
    batch_size = 3
    value = random.randint(0, 1000)
    pipe = get_dali_pipe(batch_size=batch_size, device_id=types.CPU_ONLY_DEVICE_ID, num_threads=1, value=value)
    with tf.device('/cpu'):
        ds = dali_tf.DALIDataset(pipe, device_id=types.CPU_ONLY_DEVICE_ID, batch_size=1, output_dtypes=tf.int32, output_shapes=[1])
    ds = iter(ds)
    data = next(ds)
    assert data == np.array([value])