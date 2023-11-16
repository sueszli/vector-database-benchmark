import jax
import numpy as np
import nvidia.dali.plugin.jax as dax
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.jax import DALIGenericIterator, data_iterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from jax.sharding import PositionalSharding, NamedSharding, PartitionSpec, Mesh
from jax.experimental import mesh_utils
from utils import get_dali_tensor_gpu, iterator_function_def
import jax.numpy as jnp
import itertools
batch_size = 4
shape = (1, 5)

def sequential_sharded_pipeline(batch_size, shape, device_id, shard_id, shard_size, multiple_outputs=False):
    if False:
        while True:
            i = 10
    'Helper to create DALI pipelines that return GPU tensors with sequential values\n    and are iterating over virtual sharded dataset.\n\n    For example setting shard_id for 2 and shard size for 8 will result in pipeline\n    that starts its iteration from the sample with value 16 since this is third\n    shard (shard_id=2) and the shard size is 8.\n\n    Args:\n        batch_size: Batch size for the pipeline.\n        shape : Shape of the output tensor.\n        device_id : Id of the device that pipeline will run on.\n        shard_id : Id of the shard for the pipeline.\n        shard_size : Size of the shard for the pipeline.\n        multiple_outputs : If True, pipeline will return multiple outputs.\n    '

    def create_numpy_sequential_tensors_callback():
        if False:
            print('Hello World!')
        shard_offset = shard_size * shard_id

        def numpy_sequential_tensors(sample_info):
            if False:
                while True:
                    i = 10
            return np.full(shape, sample_info.idx_in_epoch + shard_offset, dtype=np.int32)
        return numpy_sequential_tensors

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=device_id)
    def sequential_pipeline_def():
        if False:
            for i in range(10):
                print('nop')
        data = fn.external_source(source=create_numpy_sequential_tensors_callback(), num_outputs=1, batch=False, dtype=types.INT32)
        data = data[0].gpu()
        if not multiple_outputs:
            return data
        return (data, data + 0.25, data + 0.5)
    return sequential_pipeline_def()

def test_dali_sequential_sharded_tensors_to_jax_sharded_array_manuall():
    if False:
        print('Hello World!')
    assert jax.device_count() > 1, 'Multigpu test requires more than one GPU'
    pipe_0 = sequential_sharded_pipeline(batch_size=batch_size, shape=shape, device_id=0, shard_id=0, shard_size=batch_size)
    pipe_0.build()
    pipe_1 = sequential_sharded_pipeline(batch_size=batch_size, shape=shape, device_id=1, shard_id=1, shard_size=batch_size)
    pipe_1.build()
    for batch_id in range(100):
        dali_tensor_gpu_0 = pipe_0.run()[0].as_tensor()
        dali_tensor_gpu_1 = pipe_1.run()[0].as_tensor()
        jax_shard_0 = dax.integration._to_jax_array(dali_tensor_gpu_0)
        jax_shard_1 = dax.integration._to_jax_array(dali_tensor_gpu_1)
        assert jax_shard_0.device() == jax.devices()[0]
        assert jax_shard_1.device() == jax.devices()[1]
        jax_array = jax.device_put_sharded([jax_shard_0, jax_shard_1], [jax_shard_0.device(), jax_shard_1.device()])
        assert jax.numpy.array_equal(jax_array.device_buffers[0], jax.numpy.stack([jax.numpy.full(shape[1:], value, np.int32) for value in range(batch_id * batch_size, (batch_id + 1) * batch_size)]))
        assert jax.numpy.array_equal(jax_array.device_buffers[1], jax.numpy.stack([jax.numpy.full(shape[1:], value, np.int32) for value in range((batch_id + 1) * batch_size, (batch_id + 2) * batch_size)]))
        assert jax_array.device_buffers[0].device() == jax_shard_0.device()
        assert jax_array.device_buffers[1].device() == jax_shard_1.device()

def test_dali_sequential_sharded_tensors_to_jax_sharded_array_iterator_multiple_outputs():
    if False:
        print('Hello World!')
    assert jax.device_count() > 1, 'Multigpu test requires more than one GPU'
    pipe_0 = sequential_sharded_pipeline(batch_size=batch_size, shape=shape, device_id=0, shard_id=0, shard_size=batch_size, multiple_outputs=True)
    pipe_1 = sequential_sharded_pipeline(batch_size=batch_size, shape=shape, device_id=1, shard_id=1, shard_size=batch_size, multiple_outputs=True)
    output_names = ['data_0', 'data_1', 'data_2']
    dali_iterator = DALIGenericIterator([pipe_0, pipe_1], output_names, size=batch_size * 10)
    for (batch_id, batch) in enumerate(dali_iterator):
        for (output_id, output_name) in enumerate(output_names):
            jax_array = batch[output_name]
            assert jax.numpy.array_equal(jax_array.device_buffers[0], jax.numpy.stack([jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32) for value in range(batch_id * batch_size, (batch_id + 1) * batch_size)]))
            assert jax.numpy.array_equal(jax_array.device_buffers[1], jax.numpy.stack([jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32) for value in range((batch_id + 1) * batch_size, (batch_id + 2) * batch_size)]))
            assert jax_array.device_buffers[0].device() == jax.devices()[0]
            assert jax_array.device_buffers[1].device() == jax.devices()[1]
    assert batch_id == 4

def run_sharding_test(sharding):
    if False:
        for i in range(10):
            print('nop')
    dali_shard_0 = get_dali_tensor_gpu(0, 1, np.int32, 0)
    dali_shard_1 = get_dali_tensor_gpu(1, 1, np.int32, 1)
    shards = [dax.integration._to_jax_array(dali_shard_0), dax.integration._to_jax_array(dali_shard_1)]
    assert shards[0].device() == jax.devices()[0]
    assert shards[1].device() == jax.devices()[1]
    dali_sharded_array = jax.make_array_from_single_device_arrays(shape=(2,), sharding=sharding, arrays=shards)
    jax_sharded_array = jax.device_put(jnp.arange(2), sharding)
    assert (dali_sharded_array == jax_sharded_array).all()
    assert len(dali_sharded_array.device_buffers) == jax.device_count()
    assert dali_sharded_array.device_buffers[0].device() == jax.devices()[0]
    assert dali_sharded_array.device_buffers[1].device() == jax.devices()[1]

def run_sharding_iterator_test(sharding):
    if False:
        while True:
            i = 10
    assert jax.device_count() > 1, 'Multigpu test requires more than one GPU'
    pipe_0 = sequential_sharded_pipeline(batch_size=batch_size, shape=shape, device_id=0, shard_id=0, shard_size=batch_size, multiple_outputs=True)
    pipe_1 = sequential_sharded_pipeline(batch_size=batch_size, shape=shape, device_id=1, shard_id=1, shard_size=batch_size, multiple_outputs=True)
    output_names = ['data_0', 'data_1', 'data_2']
    dali_iterator = DALIGenericIterator([pipe_0, pipe_1], output_names, size=batch_size * 10, sharding=sharding)
    for (batch_id, batch) in enumerate(dali_iterator):
        for (output_id, output_name) in enumerate(output_names):
            jax_array = batch[output_name]
            assert jax.numpy.array_equal(jax_array, jax.numpy.stack([jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32) for value in range(batch_id * batch_size, (batch_id + 2) * batch_size)]))
            assert jax_array.device_buffers[0].device() == jax.devices()[0]
            assert jax_array.device_buffers[1].device() == jax.devices()[1]
    assert batch_id == 4

def test_positional_sharding_workflow():
    if False:
        return 10
    sharding = PositionalSharding(jax.devices())
    run_sharding_test(sharding)

def test_named_sharding_workflow():
    if False:
        return 10
    mesh = Mesh(jax.devices(), axis_names='device')
    sharding = NamedSharding(mesh, PartitionSpec('device'))
    run_sharding_test(sharding)

def test_positional_sharding_workflow_with_iterator():
    if False:
        return 10
    mesh = mesh_utils.create_device_mesh((jax.device_count(), 1))
    sharding = PositionalSharding(mesh)
    run_sharding_iterator_test(sharding)

def test_named_sharding_workflow_with_iterator():
    if False:
        while True:
            i = 10
    mesh = Mesh(jax.devices(), axis_names='batch')
    sharding = NamedSharding(mesh, PartitionSpec('batch'))
    run_sharding_iterator_test(sharding)

def run_sharded_iterator_test(iterator, num_iters=11):
    if False:
        i = 10
        return i + 15
    assert jax.device_count() == 2, 'Sharded iterator test requires exactly 2 GPUs'
    batch_size_per_gpu = batch_size // jax.device_count()
    assert iterator.size == 23
    for (batch_id, batch) in itertools.islice(enumerate(iterator), num_iters):
        jax_array = batch['tensor']
        sample_id = 0
        for device_id in range(jax.device_count()):
            for i in range(batch_size_per_gpu):
                ground_truth = jax.numpy.full(1, batch_id * batch_size_per_gpu + i + device_id * iterator.size, np.int32)
                assert jax.numpy.array_equal(jax_array[sample_id], ground_truth)
                sample_id += 1
        assert jax_array.device_buffers[0].device() == jax.devices()[0]
        assert jax_array.device_buffers[1].device() == jax.devices()[1]
    assert batch_id == num_iters - 1

def test_named_sharding_with_iterator_decorator():
    if False:
        i = 10
        return i + 15
    mesh = Mesh(jax.devices(), axis_names='batch')
    sharding = NamedSharding(mesh, PartitionSpec('batch'))
    output_map = ['tensor']

    @data_iterator(output_map=output_map, sharding=sharding, last_batch_policy=LastBatchPolicy.DROP, reader_name='reader')
    def iterator_function(shard_id, num_shards):
        if False:
            i = 10
            return i + 15
        return iterator_function_def(shard_id=shard_id, num_shards=num_shards)
    data_iterator_instance = iterator_function(batch_size=batch_size, num_threads=4)
    run_sharded_iterator_test(data_iterator_instance)

def test_positional_sharding_with_iterator_decorator():
    if False:
        for i in range(10):
            print('nop')
    mesh = mesh_utils.create_device_mesh((jax.device_count(), 1))
    sharding = PositionalSharding(mesh)
    output_map = ['tensor']

    @data_iterator(output_map=output_map, sharding=sharding, last_batch_policy=LastBatchPolicy.DROP, reader_name='reader')
    def iterator_function(shard_id, num_shards):
        if False:
            for i in range(10):
                print('nop')
        return iterator_function_def(shard_id=shard_id, num_shards=num_shards)
    data_iterator_instance = iterator_function(batch_size=batch_size, num_threads=4)
    run_sharded_iterator_test(data_iterator_instance)

def test_dali_sequential_iterator_decorator_non_default_device():
    if False:
        return 10

    @data_iterator(output_map=['data'], reader_name='reader')
    def iterator_function():
        if False:
            while True:
                i = 10
        return iterator_function_def()
    iter = iterator_function(num_threads=4, device_id=1, batch_size=batch_size)
    batch = next(iter)
    assert batch['data'].device_buffers[0].device() == jax.devices()[1]