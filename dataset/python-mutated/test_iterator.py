import numpy as np
import jax
import jax.numpy
import jax.dlpack
from utils import iterator_function_def
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nose_utils import raises
import itertools
batch_size = 3

def run_and_assert_sequential_iterator(iter, num_iters=4):
    if False:
        while True:
            i = 10
    'Run the iterator and assert that the output is as expected'
    for (batch_id, data) in itertools.islice(enumerate(iter), num_iters):
        jax_array = data['data']
        assert jax_array.device() == jax.devices()[0]
        for i in range(batch_size):
            assert jax.numpy.array_equal(jax_array[i], jax.numpy.full(1, batch_id * batch_size + i, np.int32))
    assert batch_id == num_iters - 1

def test_dali_sequential_iterator():
    if False:
        return 10
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    iter = DALIGenericIterator([pipe], ['data'], reader_name='reader')
    run_and_assert_sequential_iterator(iter)

@raises(AssertionError, glob='JAX iterator does not support partial last batch policy.')
def test_iterator_last_batch_policy_partial_exception():
    if False:
        return 10
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    DALIGenericIterator([pipe], ['data'], reader_name='reader', last_batch_policy=LastBatchPolicy.PARTIAL)