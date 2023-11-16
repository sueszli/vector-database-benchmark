import numpy as np
import nvidia.dali.fn as fn
from nose.plugins.attrib import attr
from nose2.tools import params
from numpy.random import default_rng
from nvidia.dali import pipeline_def
from nvidia.dali.tensors import TensorCPU, TensorGPU
from nose_utils import raises
from test_utils import to_array
max_batch_size = 256
max_test_value = 10000000.0

@pipeline_def(batch_size=max_batch_size, num_threads=1, device_id=0)
def identity_pipe(use_copy_kernel, blocking):
    if False:
        print('Hello World!')
    ins = (fn.external_source(name='numpy', device='cpu', use_copy_kernel=use_copy_kernel, blocking=blocking, cycle=False, cuda_stream=1, no_copy=True, batch=True, batch_info=False, parallel=False), fn.external_source(name='cupy', device='gpu', use_copy_kernel=use_copy_kernel, blocking=blocking, cycle=False, cuda_stream=1, no_copy=True, batch=True, batch_info=False, parallel=False), fn.external_source(name='torch_cpu', device='cpu', use_copy_kernel=use_copy_kernel, blocking=blocking, cycle=False, cuda_stream=1, no_copy=True, batch=True, batch_info=False, parallel=False), fn.external_source(name='torch_gpu', device='gpu', use_copy_kernel=use_copy_kernel, blocking=blocking, cycle=False, cuda_stream=1, no_copy=True, batch=True, batch_info=False, parallel=False), fn.external_source(name='tensor_cpu', device='cpu', use_copy_kernel=use_copy_kernel, blocking=blocking, cycle=False, cuda_stream=1, no_copy=True, batch=True, batch_info=False, parallel=False), fn.external_source(name='tensor_gpu', device='gpu', use_copy_kernel=use_copy_kernel, blocking=blocking, cycle=False, cuda_stream=1, no_copy=True, batch=True, batch_info=False, parallel=False), fn.external_source(name='list_cpu', device='cpu', use_copy_kernel=use_copy_kernel, blocking=blocking, cycle=False, cuda_stream=1, no_copy=True, batch=True, batch_info=False, parallel=False))
    return tuple((i.gpu() for i in ins))

@attr('torch')
@attr('cupy')
@params((True, True), (False, True), (True, False), (False, False))
def test_pipeline_inputs_prefetch_queue_depth(use_copy_kernel, blocking):
    if False:
        return 10
    import torch
    import cupy as cp
    rng = default_rng()
    n_iterations = 8
    p = identity_pipe(use_copy_kernel, blocking, prefetch_queue_depth=1)
    p.build()
    for _ in range(n_iterations):
        batch_size = rng.integers(1, max_batch_size)
        random_in = rng.random(size=(batch_size, 4, 6, 2))
        in_list_cpu = [rng.integers(low=-max_test_value, high=max_test_value, size=(5, 3, 2)) for _ in range(batch_size)]
        (numpy, cupy, torch_cpu, torch_gpu, tensor_cpu, tensor_gpu, out_list_cpu) = p.run(numpy=random_in, cupy=cp.array(random_in), torch_cpu=torch.Tensor(random_in), torch_gpu=torch.Tensor(random_in).cuda(), tensor_cpu=TensorCPU(random_in), tensor_gpu=TensorGPU(cp.array(random_in)), list_cpu=in_list_cpu)
        assert np.all(np.isclose(to_array(numpy), random_in))
        assert np.all(np.isclose(to_array(cupy), random_in))
        assert np.all(np.isclose(to_array(torch_cpu), random_in))
        assert np.all(np.isclose(to_array(torch_gpu), random_in))
        assert np.all(np.isclose(to_array(tensor_cpu), random_in))
        assert np.all(np.isclose(to_array(tensor_gpu), random_in))
        for (ref, tst) in zip(in_list_cpu, out_list_cpu):
            assert np.all(np.isclose(to_array(tst), ref))

@attr('torch')
@attr('cupy')
@params((True, True), (False, True), (True, False), (False, False))
def test_pipeline_inputs_exec_pipelined(use_copy_kernel, blocking):
    if False:
        print('Hello World!')
    import torch
    import cupy as cp
    rng = default_rng()
    n_iterations = 8
    p = identity_pipe(use_copy_kernel, blocking, exec_pipelined=False, exec_async=False)
    p.build()
    for _ in range(n_iterations):
        batch_size = rng.integers(1, max_batch_size)
        random_in = rng.random(size=(batch_size, 4, 6, 2))
        in_list_cpu = [rng.integers(low=-max_test_value, high=max_test_value, size=(5, 3, 2)) for _ in range(batch_size)]
        (numpy, cupy, torch_cpu, torch_gpu, tensor_cpu, tensor_gpu, out_list_cpu) = p.run(numpy=random_in, cupy=cp.array(random_in), torch_cpu=torch.Tensor(random_in), torch_gpu=torch.Tensor(random_in).cuda(), tensor_cpu=TensorCPU(random_in), tensor_gpu=TensorGPU(cp.array(random_in)), list_cpu=in_list_cpu)
        assert np.all(np.isclose(to_array(numpy), random_in))
        assert np.all(np.isclose(to_array(cupy), random_in))
        assert np.all(np.isclose(to_array(torch_cpu), random_in))
        assert np.all(np.isclose(to_array(torch_gpu), random_in))
        assert np.all(np.isclose(to_array(tensor_cpu), random_in))
        assert np.all(np.isclose(to_array(tensor_gpu), random_in))
        for (ref, tst) in zip(in_list_cpu, out_list_cpu):
            assert np.all(np.isclose(to_array(tst), ref))

@raises(RuntimeError, glob='*`prefetch_queue_depth` in Pipeline constructor shall be set to 1*')
def test_incorrect_prefetch_queue_depth():
    if False:
        while True:
            i = 10
    p = identity_pipe(False, False)
    p.build()
    rng = default_rng()
    batch_size = rng.integers(1, max_batch_size)
    random_in = rng.random(size=(batch_size, 4, 6, 2))
    p.run(numpy=random_in)