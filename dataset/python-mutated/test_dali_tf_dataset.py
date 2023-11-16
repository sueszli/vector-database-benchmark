import tensorflow as tf
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.plugin.tf as dali_tf
from nose_utils import raises
from test_utils_tensorflow import get_image_pipeline

@raises(ValueError, "Two structures don't have the same sequence length*length 3*length 2")
def test_different_num_shapes_dtypes():
    if False:
        for i in range(10):
            print('nop')
    batch_size = 12
    num_threads = 4
    (dataset_pipe, shapes, dtypes) = get_image_pipeline(batch_size, num_threads, 'cpu')
    dtypes = tuple(dtypes[0:2])
    with tf.device('/cpu:0'):
        dali_tf.DALIDataset(pipeline=dataset_pipe, batch_size=batch_size, output_shapes=shapes, output_dtypes=dtypes, num_threads=num_threads)

@raises(RuntimeError, 'some operators*cannot be used with TensorFlow Dataset API and DALIIterator')
def test_python_operator_not_allowed_in_tf_dataset_error():
    if False:
        while True:
            i = 10
    pipeline = Pipeline(1, 1, 0, exec_pipelined=False, exec_async=False)
    with pipeline:
        output = fn.python_function(function=lambda : np.zeros((3, 3, 3)))
        pipeline.set_outputs(output)
    shapes = (1, 3, 3, 3)
    dtypes = tf.float32
    with tf.device('/cpu:0'):
        _ = dali_tf.DALIDataset(pipeline=pipeline, batch_size=1, output_shapes=shapes, output_dtypes=dtypes, num_threads=1, device_id=0)