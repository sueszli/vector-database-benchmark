import tensorflow as tf
import numpy as np
from nvidia.dali.pipeline.experimental import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
from nose.tools import with_setup
from test_utils_tensorflow import skip_inputs_for_incompatible_tf

@with_setup(skip_inputs_for_incompatible_tf)
def test_both_tf_and_dali_conditionals():
    if False:
        i = 10
        return i + 15

    @pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0)
    def dali_conditional_pipeline():
        if False:
            return 10
        iter_id = fn.external_source(source=lambda x: np.array(x.iteration), batch=False)
        if iter_id & 1 == 0:
            output = types.Constant(np.array(-1), device='cpu')
        else:
            output = types.Constant(np.array(1), device='cpu')
        return output
    with tf.device('/cpu:0'):
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(pipeline=dali_conditional_pipeline(), batch_size=5, output_shapes=(5,), output_dtypes=tf.int32, num_threads=4, device_id=0)

        @tf.function
        def tf_function_with_conditionals(dali_dataset):
            if False:
                return 10
            negative = tf.constant(0)
            positive = tf.constant(0)
            for input in dali_dataset:
                if tf.reduce_sum(input) < 0:
                    negative = negative + 1
                else:
                    positive = positive + 1
            return (negative, positive)
        (pos, neg) = tf_function_with_conditionals(dali_dataset.take(5))
        assert pos == 3
        assert neg == 2