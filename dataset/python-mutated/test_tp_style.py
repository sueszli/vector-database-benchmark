import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import ColwiseParallel, make_input_replicate_1d, make_input_reshard_replicate, make_input_shard_1d, make_output_replicate_1d, make_output_reshard_tensor, make_output_shard_1d, make_output_tensor, PrepareModuleInput, PrepareModuleOutput, RowwiseParallel
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms, NUM_DEVICES

class TensorParallelStyleTest(DTensorTestBase):

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        return NUM_DEVICES

    def _1d_input_func_check(self, input_local_tensor, expected_local_tensor, func, error_msgs='device_mesh is not passed nor can be inferred') -> None:
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, error_msgs):
            dtensor = func(input_local_tensor)
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        dtensor = func(input_local_tensor, device_mesh)
        result = dtensor[0] if isinstance(dtensor, tuple) else dtensor
        self.assertEqual(expected_local_tensor, result.to_local())
        dtensor = func(dtensor)
        result = dtensor[0] if isinstance(dtensor, tuple) else dtensor
        self.assertEqual(expected_local_tensor, result.to_local())
        dtensor = func(dtensor, device_mesh)
        result = dtensor[0] if isinstance(dtensor, tuple) else dtensor
        self.assertEqual(expected_local_tensor, result.to_local())

    @with_comms
    def test_make_input_replicate_1d(self):
        if False:
            i = 10
            return i + 15
        tensor = torch.rand(8, 16, device=self.device_type)
        self._1d_input_func_check(tensor, tensor, make_input_replicate_1d)

    @with_comms
    def test_make_input_shard_1d(self):
        if False:
            while True:
                i = 10
        tensor = torch.rand(8, 16, device=self.device_type)
        self._1d_input_func_check(tensor, tensor, make_input_shard_1d)

    @with_comms
    def test_make_input_reshard_replicate(self):
        if False:
            for i in range(10):
                print('nop')
        tensor = torch.rand(8, 16, device=self.device_type)
        gathered_tensor = [torch.empty(8, 16, device=self.device_type) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor)
        self._1d_input_func_check(tensor, gathered_tensor, make_input_reshard_replicate)

    def _test_prepare_output(self, func, spec, dim=None, device_mesh_input_none=False):
        if False:
            print('Hello World!')
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        tensor = torch.rand(8, 16, device=self.device_type)
        dtensor = distribute_tensor(tensor, device_mesh, spec)
        device_mesh_input = None if device_mesh_input_none else device_mesh
        if dim is not None:
            output = func(dtensor, device_mesh_input, dim)
        else:
            output = func(dtensor, device_mesh_input)
        return (output, dtensor, device_mesh)

    @with_comms
    def test_make_output_shard_1d(self):
        if False:
            while True:
                i = 10
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_shard_1d, [Shard(0)], 1)
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(1)]))
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_shard_1d, [Replicate()], 0)
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]))
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_shard_1d, [Shard(0)], 1, True)
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(1)]))

    @with_comms
    def test_make_output_replicate_1d(self):
        if False:
            while True:
                i = 10
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_replicate_1d, [Shard(0)])
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]))
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_replicate_1d, [Shard(0)], None, True)
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]))

    @with_comms
    def test_make_output_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_tensor, [Shard(0)])
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]).to_local())
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_tensor, [Replicate()])
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]).to_local())
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_tensor, [Shard(0)], None, True)
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]).to_local())

    @with_comms
    def test_make_output_reshard_tensor(self):
        if False:
            while True:
                i = 10
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_reshard_tensor, [Shard(0)])
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local())
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_reshard_tensor, [Replicate()])
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local())
        (output, dtensor, device_mesh) = self._test_prepare_output(make_output_reshard_tensor, [Shard(0)], None, True)
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local())

    def _test_prepare_output_error(self, func):
        if False:
            print('Hello World!')
        tensor = torch.rand(8, 16, device=self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        output = [dtensor]
        with self.assertRaisesRegex(AssertionError, f'Expect output of Tensor Parallel to be a DTensor, but found {type(output)}.'):
            func(output, device_mesh)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(self.world_size // 2, 2))
        with self.assertRaisesRegex(AssertionError, 'device_mesh has dims 2 but expected to be 1 for output.'):
            func(dtensor, device_mesh)

    def _test_prepare_output_error_new(self, func):
        if False:
            for i in range(10):
                print('nop')
        tensor = torch.rand(8, 16, device=self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        output = [dtensor]
        with self.assertRaisesRegex(RuntimeError, f'Tensor parallel module expects DTensor or tensor when layout specified but received {type(output)}!'):
            func(output, device_mesh)

    @with_comms
    def test_prepare_output_error(self):
        if False:
            while True:
                i = 10
        self._test_prepare_output_error(make_output_shard_1d)
        self._test_prepare_output_error(make_output_replicate_1d)
        self._test_prepare_output_error(make_output_tensor)

    @with_comms
    def test_rowwise_parallel_style(self):
        if False:
            i = 10
            return i + 15
        tensor = torch.rand(8, 16, device=self.device_type)
        rs = RowwiseParallel()
        self._1d_input_func_check([tensor], tensor, rs._prepare_input, error_msgs='No device mesh is currently active')
        (output, dtensor, device_mesh) = self._test_prepare_output(rs._prepare_output, [Shard(0)])
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]).to_local())
        (output, dtensor, device_mesh) = self._test_prepare_output(rs._prepare_output, [Shard(0)], None, True)
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]).to_local())
        self._test_prepare_output_error_new(rs._prepare_output)

    @with_comms
    def test_colwise_parallel_style(self):
        if False:
            return 10
        tensor = torch.rand(8, 16, device=self.device_type)
        cs = ColwiseParallel()
        self._1d_input_func_check([tensor], tensor, cs._prepare_input, error_msgs='No device mesh is currently active')
        (output, dtensor, device_mesh) = self._test_prepare_output(cs._prepare_output, [Shard(-1)])
        self.assertEqual(output, dtensor.to_local())

    @with_comms
    def test_prepare_module_input(self):
        if False:
            i = 10
            return i + 15
        tensor = torch.rand(8, 16, device=self.device_type)
        gathered_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, tensor)
        gathered_tensors = torch.cat(gathered_tensors, dim=0).contiguous()
        prepare_hook = PrepareModuleInput(input_layouts=[Shard(0)], output_layouts=[Replicate()])
        self._1d_input_func_check([tensor], gathered_tensors, prepare_hook._prepare_input, error_msgs='No device mesh is currently active')

    @with_comms
    def test_prepare_module_input_multiple_inputs(self):
        if False:
            return 10

        class TestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return self.linear(x) + y
        test_mod = TestModule().to(self.device_type)
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        parallelize_module(test_mod.linear, mesh, ColwiseParallel())
        parallelize_module(test_mod, mesh, PrepareModuleInput(input_layouts=(Shard(0), None), output_layouts=(Replicate(), None)))
        output = test_mod(torch.randn(2, 8, device=self.device_type), torch.ones(self.world_size * 2, 8 // self.world_size, device=self.device_type))
        self.assertEqual(output.shape, (self.world_size * 2, 8 // self.world_size))

    @with_comms
    def test_prepare_module_output(self):
        if False:
            while True:
                i = 10
        tensor = torch.rand(8, 16, device=self.device_type)
        prepare_hook = PrepareModuleOutput(input_layouts=[Replicate()], output_layouts=[Shard(0)])
        (output, dtensor, device_mesh) = self._test_prepare_output(prepare_hook._prepare_output, [Replicate()])
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local())
if __name__ == '__main__':
    run_tests()