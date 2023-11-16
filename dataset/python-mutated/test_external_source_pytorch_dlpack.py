import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import torch
from nvidia.dali.backend_impl import *
from nvidia.dali import Pipeline
from torch.utils.dlpack import to_dlpack, from_dlpack
from test_utils import check_output

class TestIterator:

    def __init__(self, n, batch_size, dims=[2], as_tensor=False, device='cuda'):
        if False:
            print('Hello World!')
        self.batch_size = batch_size
        self.dims = dims
        self.n = n
        self.as_tensor = as_tensor
        self.i = 0
        self.device = device

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.n

    def __iter__(self):
        if False:
            return 10
        return TestIterator(self.n, self.batch_size, self.dims, self.as_tensor, self.device)

    def __next__(self):
        if False:
            print('Hello World!')
        np.random.seed(12345 * self.i + 4321)
        torch.random.manual_seed(12345 * self.i + 4321)

        def generate(dim):
            if False:
                for i in range(10):
                    print('nop')
            shape = np.random.randint(1, 10, [dim]).tolist()
            if self.as_tensor:
                data = to_dlpack(torch.rand(size=[self.batch_size] + shape, device=self.device))
            else:
                data = [to_dlpack(torch.rand(shape, device=self.device)) for _ in range(self.batch_size)]
            return data
        if self.i < self.n:
            self.i += 1
            if isinstance(self.dims, (list, tuple)):
                return [generate(d) for d in self.dims]
            else:
                return generate(self.dims)
        else:
            self.i = 0
            raise StopIteration
    next = __next__

def asnumpy(x, device):
    if False:
        print('Hello World!')
    if x is None:
        return None
    if isinstance(x, list):
        return [asnumpy(y, device) for y in x]
    if isinstance(x, np.ndarray):
        return x
    if device == 'cpu':
        return from_dlpack(x).numpy()
    else:
        return from_dlpack(x).cpu().numpy()

def run_and_check(pipe, ref_iterable):
    if False:
        while True:
            i = 10
    iter_ref = iter(ref_iterable)
    i = 0
    while True:
        try:
            pipe_out = pipe.run()
            data = next(iter_ref)
            data = asnumpy(data, iter_ref.device)
            check_output(pipe_out, data)
            i += 1
        except StopIteration:
            break
    assert i == len(ref_iterable)

def _test_iter_setup(use_fn_api, by_name, src_device, gen_device):
    if False:
        for i in range(10):
            print('nop')
    batch_size = 7

    class IterSetupPipeline(Pipeline):

        def __init__(self, iterator, num_threads, device_id, src_device):
            if False:
                return 10
            super().__init__(batch_size=iterator.batch_size, num_threads=num_threads, device_id=device_id)
            self.iterator = iterator
            self._device = src_device

        def define_graph(self):
            if False:
                return 10
            if use_fn_api:
                self.batch_1 = fn.external_source(device=self._device, name='src1', cuda_stream=torch.cuda.default_stream())
                self.batch_2 = fn.external_source(device=self._device, name='src2', cuda_stream=torch.cuda.default_stream())
            else:
                input_1 = ops.ExternalSource(device=self._device)
                input_2 = ops.ExternalSource(device=self._device)
                self.batch_1 = input_1(name='src1')
                self.batch_2 = input_2(name='src2')
            return [self.batch_1, self.batch_2]

        def iter_setup(self):
            if False:
                while True:
                    i = 10
            (batch_1, batch_2) = next(self.iterator)
            if by_name:
                self.feed_input('src1', batch_1, cuda_stream=torch.cuda.default_stream())
                self.feed_input('src2', batch_2, cuda_stream=torch.cuda.default_stream())
            else:
                self.feed_input(self.batch_1, batch_1, cuda_stream=torch.cuda.default_stream())
                self.feed_input(self.batch_2, batch_2, cuda_stream=torch.cuda.default_stream())
    iter_num = 5
    source = TestIterator(n=iter_num, batch_size=batch_size, dims=[2, 3], device=gen_device)
    pipe = IterSetupPipeline(iter(source), 3, 0, src_device)
    pipe.build()
    run_and_check(pipe, source)

def test_iter_setup():
    if False:
        for i in range(10):
            print('nop')
    for use_fn_api in [False, True]:
        for by_name in [False, True]:
            for src_device in ['cpu', 'gpu']:
                for gen_device in ['cpu', 'cuda']:
                    yield (_test_iter_setup, use_fn_api, by_name, src_device, gen_device)

def _test_external_source_callback_torch_stream(src_device, gen_device):
    if False:
        while True:
            i = 10
    with torch.cuda.stream(torch.cuda.Stream()):
        for attempt in range(10):
            t0 = torch.tensor([attempt * 100 + 1.5], dtype=torch.float32, device=gen_device)
            increment = torch.tensor([10], dtype=torch.float32, device=gen_device)
            pipe = Pipeline(1, 3, 0)

            def gen_batch():
                if False:
                    while True:
                        i = 10
                nonlocal t0
                t0 += increment
                return [to_dlpack(t0)]
            pipe.set_outputs(fn.external_source(source=gen_batch, device=src_device, cuda_stream=torch.cuda.current_stream()))
            pipe.build()
            for i in range(10):
                check_output(pipe.run(), [np.array([attempt * 100 + (i + 1) * 10 + 1.5], dtype=np.float32)])

def test_external_source_callback_torch_stream():
    if False:
        print('Hello World!')
    for src_device in ['cpu', 'gpu']:
        for gen_device in ['cpu', 'cuda']:
            yield (_test_external_source_callback_torch_stream, src_device, gen_device)