import itertools
import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import _Partial, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

class RedistributeTest(DTensorTestBase):

    @with_comms
    def test_shard_to_replicate_forward_backward(self):
        if False:
            return 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]
        input_sizes_and_shard_dim = [((self.world_size * 3, 3), 0), ((self.world_size * 3 + 1, 3), 0), ((self.world_size * 3 + 2, 3), 0), ((3, self.world_size * 3), 1), ((3, self.world_size * 3 + 1), 1), ((3, self.world_size * 3 + 2), 1)]
        for (input_size, shard_dim) in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]
            expected_tensor = torch.randn(input_size, device=self.device_type, requires_grad=True)
            dtensor = distribute_tensor(expected_tensor, device_mesh, shard_spec)
            reshard_dtensor = dtensor.redistribute(device_mesh, replica_spec)
            self.assertEqual(reshard_dtensor.size(), torch.Size(input_size))
            self.assertEqual(expected_tensor, reshard_dtensor.to_local())
            grad_output = torch.ones_like(reshard_dtensor)
            reshard_dtensor.backward(grad_output)
            grad_input = dtensor.grad
            self.assertEqual(grad_input.placements, shard_spec)
            self.assertEqual(grad_input.to_local(), torch.ones(dtensor.to_local().size()))

    @with_comms
    def test_replicate_to_replicate_forward_backward(self):
        if False:
            return 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        replica_tensor = distribute_tensor(local_tensor, device_mesh, replica_spec)
        reshard_replica_tensor = replica_tensor.redistribute(device_mesh, replica_spec)
        self.assertEqual(replica_tensor.size(), local_tensor.size())
        self.assertEqual(replica_tensor, reshard_replica_tensor)
        grad_output = torch.ones_like(reshard_replica_tensor)
        reshard_replica_tensor.backward(grad_output)
        grad_input = replica_tensor.grad
        self.assertEqual(grad_input.placements, replica_spec)
        self.assertEqual(grad_input.to_local(), torch.ones(12, 3))

    @with_comms
    def test_replicate_to_shard_forward_backward(self):
        if False:
            print('Hello World!')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]
        input_sizes_and_shard_dim = [((self.world_size * 3, 3), 0), ((self.world_size * 3 + 1, 3), 0), ((self.world_size * 3 + 2, 3), 0), ((3, self.world_size * 3), 1), ((3, self.world_size * 3 + 1), 1), ((3, self.world_size * 3 + 2), 1)]
        for (input_size, shard_dim) in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]
            local_replica = torch.randn(input_size, device=self.device_type, requires_grad=True)
            splitted_list = list(torch.chunk(local_replica, self.world_size, dim=shard_dim))
            local_tensor = splitted_list[self.rank]
            replica_tensor = distribute_tensor(local_replica, device_mesh, replica_spec)
            reshard_tensor = replica_tensor.redistribute(device_mesh, shard_spec)
            self.assertEqual(reshard_tensor.size(), replica_tensor.size())
            self.assertEqual(reshard_tensor.placements, shard_spec)
            self.assertEqual(reshard_tensor.to_local(), local_tensor)
            grad_output = torch.ones_like(reshard_tensor)
            reshard_tensor.backward(grad_output)
            grad_input = replica_tensor.grad
            self.assertEqual(grad_input.placements, replica_spec)
            self.assertEqual(grad_input.to_local(), torch.ones(input_size))

    @with_comms
    def test_partial_to_replicate_forward_backward(self):
        if False:
            for i in range(10):
                print('nop')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_local = torch.ones(12, 3, device=self.device_type, requires_grad=True)
        partial_spec = [_Partial()]
        replica_spec = [Replicate()]
        partial_tensor = DTensor.from_local(partial_local, device_mesh, partial_spec)
        global_partial_tensor = partial_tensor.redistribute(device_mesh, replica_spec)
        self.assertEqual(partial_tensor.size(), partial_local.size())
        self.assertEqual(partial_local * self.world_size, global_partial_tensor.to_local())
        global_partial_tensor.backward(torch.ones_like(global_partial_tensor))
        self.assertIsNotNone(partial_local.grad)
        self.assertEqual(partial_local.grad, torch.ones_like(partial_local) / self.world_size)

    @with_comms
    def test_replicate_to_partial(self):
        if False:
            while True:
                i = 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        partial_spec = _Partial()
        replica_spec = Replicate()
        replica_tensor = distribute_tensor(local_tensor, device_mesh, [replica_spec])
        with self.assertRaisesRegex(RuntimeError, 'Can not redistribute to _Partial'):
            partial_tensor = replica_tensor.redistribute(device_mesh, [partial_spec])
        from torch.distributed._tensor.redistribute import Redistribute
        partial_tensor = Redistribute.apply(replica_tensor, device_mesh, [partial_spec])
        self.assertEqual(partial_tensor.size(), local_tensor.size())
        self.assertEqual(replica_tensor.to_local() / self.world_size, partial_tensor.to_local())
        local_tensor = torch.randn(12, 3, device=self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(self.world_size // 2, 2))
        replica_tensor = distribute_tensor(local_tensor, device_mesh, [replica_spec, replica_spec])
        partial_tensor = Redistribute.apply(replica_tensor, device_mesh, [partial_spec, partial_spec])
        self.assertEqual(partial_tensor.size(), local_tensor.size())
        self.assertEqual(replica_tensor.to_local() / self.world_size, partial_tensor.to_local())

    @with_comms
    def test_partial_to_shard(self):
        if False:
            print('Hello World!')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_spec = [_Partial()]
        my_rank = device_mesh.get_rank()
        input_sizes_and_shard_dim = [((self.world_size * 3, 3), 0), ((self.world_size * 3 + 1, 3), 0), ((self.world_size * 3 + 2, 3), 0), ((3, self.world_size * 3), 1), ((3, self.world_size * 3 + 1), 1), ((3, self.world_size * 3 + 2), 1)]
        for (input_size, shard_dim) in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]
            partial_local = torch.ones(input_size, device=self.device_type)
            partial_tensor = DTensor.from_local(partial_local, device_mesh, partial_spec, run_check=False)
            full_chunk_size = (input_size[shard_dim] + self.world_size - 1) // self.world_size
            chunk_sizes = [max(min(input_size[shard_dim], full_chunk_size * (idx + 1)) - full_chunk_size * idx, 0) for idx in range(self.world_size)]
            local_shape = list(input_size)
            local_shape[shard_dim] = chunk_sizes[my_rank]
            scatter_shard_tensor = partial_tensor.redistribute(device_mesh, shard_spec)
            self.assertEqual(scatter_shard_tensor.size(), partial_tensor.size())
            self.assertEqual(scatter_shard_tensor.placements, shard_spec)
            self.assertEqual(scatter_shard_tensor.to_local(), torch.ones(local_shape) * self.world_size)

    @with_comms
    def test_redistribute_negative_shard_dim(self):
        if False:
            i = 10
            return i + 15
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        shard_spec = [Shard(1)]
        shard_minus_spec = [Shard(-1)]
        shard_tensor = distribute_tensor(local_tensor, device_mesh, shard_spec)
        self.assertEqual(shard_tensor.placements[0].dim, 1)
        reshard_tensor = shard_tensor.redistribute(device_mesh, shard_minus_spec)
        self.assertEqual(shard_tensor.placements[0].dim, 1)

class MultiDimRedistributeTest(DTensorTestBase):

    @property
    def world_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return 8

    @with_comms
    def test_multi_dim_mesh(self):
        if False:
            return 10
        devices = torch.arange(self.world_size)
        for mesh_shape in [devices, devices.view(4, 2), devices.view(2, 2, 2)]:
            mesh_shape = torch.arange(self.world_size).view(-1, 2)
            device_mesh = DeviceMesh(self.device_type, mesh_shape)
            tensor_shape = (16, 24)
            if torch.distributed.get_rank() == 0:
                full_tensor = torch.randn(*tensor_shape)
            else:
                full_tensor = torch.ones(*tensor_shape)
            possibilities = [Replicate()] + [Shard(i) for i in range(full_tensor.ndim)]
            all_outputs = list(itertools.product(*mesh_shape.ndim * [possibilities]))
            all_inputs = list(itertools.product(*mesh_shape.ndim * [possibilities + [_Partial()]]))
            for inputs in all_inputs:
                repl_inputs = [Replicate() if s.is_partial() else s for s in inputs]
                dt = distribute_tensor(full_tensor, device_mesh, repl_inputs)
                if repl_inputs != inputs:
                    dt = DTensor.from_local(dt.to_local(), device_mesh, inputs, run_check=False)
                for outputs in all_outputs:
                    dt2 = dt.redistribute(device_mesh, outputs)
                    local_full = dt2.full_tensor()
                    if torch.distributed.get_rank() == 0:
                        self.assertEqual(local_full.shape, full_tensor.shape)
                        num_sums = 1
                        for (idx, input) in enumerate(inputs):
                            if input.is_partial():
                                num_sums *= mesh_shape.size(idx)
                        expected = num_sums * full_tensor
                        self.assertEqual(local_full, expected)
if __name__ == '__main__':
    run_tests()