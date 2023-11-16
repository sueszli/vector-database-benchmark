import numpy as np
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as types
import random
import tensorflow as tf
from nvidia.dali.pipeline import pipeline_def
try:
    from tensorflow.compat.v1 import Session
except Exception:
    from tensorflow import Session

@pipeline_def()
def get_dali_pipe(value):
    if False:
        for i in range(10):
            print('nop')
    data = types.Constant(value)
    return data

def get_data(batch_size, value):
    if False:
        return 10
    pipe = get_dali_pipe(batch_size=batch_size, device_id=types.CPU_ONLY_DEVICE_ID, num_threads=1, value=value)
    daliop = dali_tf.DALIIterator()
    out = []
    with tf.device('/cpu'):
        data = daliop(pipeline=pipe, shapes=[batch_size], dtypes=[tf.int32], device_id=types.CPU_ONLY_DEVICE_ID)
        out.append(data)
    return [out]

def test_dali_tf_op_cpu_only():
    if False:
        for i in range(10):
            print('nop')
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass
    value = random.randint(0, 1000)
    batch_size = 3
    test_batch = get_data(batch_size, value)
    with Session() as sess:
        data = sess.run(test_batch)
        assert (data == np.array([value] * batch_size)).all()