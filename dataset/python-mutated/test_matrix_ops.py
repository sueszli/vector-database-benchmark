import itertools
from typing import cast, List, Optional
import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import _Partial, Placement, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, skip_unless_torch_gpu, with_comms

class DistMatrixOpsTest(DTensorTestBase):

    @with_comms
    def test_addmm(self):
        if False:
            i = 10
            return i + 15
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]
        tensor_to_shard = torch.randn(12, 8)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        tensor_to_replicate = torch.randn(8, 4)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
        input_tensor = torch.randn(4)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)
        dist_res = torch.addmm(input, mat1, mat2)
        local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
        self.assertEqual(dist_res.full_tensor(), local_res)

    @with_comms
    def test_addmm_auto_redistribute(self):
        if False:
            i = 10
            return i + 15
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = [Shard(0)]
        shard1_spec = [Shard(1)]
        replica_spec = [Replicate()]
        tensor_to_shard1 = torch.randn(12, 8, requires_grad=True)
        mat1 = distribute_tensor(tensor_to_shard1, device_mesh, shard1_spec)
        tensor_to_shard0 = torch.randn(8, 4, requires_grad=True)
        mat2 = distribute_tensor(tensor_to_shard0, device_mesh, shard0_spec)
        input_tensor = torch.randn(4, requires_grad=True)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)
        local_res = torch.addmm(input_tensor, tensor_to_shard1, tensor_to_shard0)
        dist_res = torch.addmm(input, mat1, mat2)
        self.assertIsInstance(dist_res, DTensor)
        self.assertIsInstance(dist_res.placements[0], _Partial)
        dist_local_res = dist_res.full_tensor()
        self.assertEqual(local_res, dist_local_res)
        dist_local_res.sum().backward()
        local_res.sum().backward()
        self.assertIsNotNone(mat2.grad)
        self.assertEqual(mat2.grad.full_tensor(), tensor_to_shard0.grad)

    @with_comms
    def test_mm(self):
        if False:
            for i in range(10):
                print('nop')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        replica_spec = Replicate()
        t1 = torch.randn(12, 8, requires_grad=True)
        t2 = torch.randn(8, 16, requires_grad=True)
        local_res = torch.mm(t1, t2)

        def test_placement_comb(placements1: List[Placement], placements2: List[Placement]) -> None:
            if False:
                print('Hello World!')
            dt1 = distribute_tensor(t1, device_mesh, placements1)
            dt2 = distribute_tensor(t2, device_mesh, placements2)
            dist_res: DTensor = cast(DTensor, torch.mm(dt1, dt2)).redistribute(device_mesh, [replica_spec])
            self.assertEqual(dist_res.to_local(), local_res)
            grad_dist_res = torch.ones_like(dist_res)
            dist_res.backward(grad_dist_res)
            self.assertIsNotNone(dt1.grad)
        placement_specs = [shard0_spec, shard1_spec, replica_spec]
        shard_specs_comb = list(itertools.product(placement_specs, placement_specs))
        for spec in shard_specs_comb:
            test_placement_comb([spec[0]], [spec[1]])

    @with_comms
    def test_t(self):
        if False:
            while True:
                i = 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        tensor_to_transpose = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_transpose, device_mesh, shard_spec)
        tranposed_mat = mat.t()
        self.assertEqual(tranposed_mat.size(), torch.Size([8, 12]))
        self.assertEqual(tranposed_mat.placements, [Shard(1)])
        tranposed_mat2 = tranposed_mat.t()
        self.assertEqual(tranposed_mat2.size(), torch.Size([12, 8]))
        self.assertEqual(tranposed_mat2.placements, shard_spec)

    @with_comms
    def test_t_partial(self):
        if False:
            print('Hello World!')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        a = torch.randn(12, 8)
        b = torch.randn(8, 4)
        c = torch.mm(a, b).t()
        da = distribute_tensor(a, device_mesh, [Shard(1)])
        db = distribute_tensor(b, device_mesh, [Shard(0)])
        dc = torch.mm(da, db).t()
        self.assertTrue(isinstance(dc.placements[0], _Partial))
        self.assertEqual(c, dc.redistribute(device_mesh, [Replicate()]).to_local())

    @with_comms
    @skip_unless_torch_gpu
    def test_baddbmm(self):
        if False:
            print('Hello World!')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        batch_1 = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        batch_2 = torch.rand(4, 8, 8, device=self.device_type, requires_grad=True)

        def test_placement_comb(tensor_placements: List[Placement], batch_1_placements: List[Placement], batch_2_placements: List[Placement], beta: int, alpha: int, batch_1_grad: Optional[torch.Tensor]) -> None:
            if False:
                print('Hello World!')
            tensor_dt = distribute_tensor(tensor, device_mesh, tensor_placements)
            batch_1_dt = distribute_tensor(batch_1, device_mesh, batch_1_placements)
            batch_2_dt = distribute_tensor(batch_2, device_mesh, batch_2_placements)
            dist_res = cast(DTensor, torch.baddbmm(tensor_dt, batch_1_dt, batch_2_dt, beta=beta, alpha=alpha)).redistribute(device_mesh, [Replicate()])
            dist_local_res = dist_res.to_local()
            assert not torch.isnan(local_result).any()
            assert not torch.isnan(dist_local_res).any()
            self.assertEqual(dist_local_res.detach(), local_result.detach())
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        shard2_spec = Shard(2)
        replica_spec = Replicate()
        shard_specs = [shard0_spec, shard1_spec, shard2_spec, replica_spec]
        shard_specs_comb = list(itertools.product(shard_specs, shard_specs, shard_specs))
        numeric_params_comb = [(0.0, 0.5), (0.8, 0.5)]
        for (beta, alpha) in numeric_params_comb:
            local_result = torch.baddbmm(tensor, batch_1, batch_2, beta=beta, alpha=alpha)
            grad_local_res = torch.ones_like(local_result)
            local_result.backward(grad_local_res)
            for spec in shard_specs_comb:
                test_placement_comb([spec[0]], [spec[1]], [spec[2]], beta, alpha, batch_1.grad)

    @with_comms
    def test_bmm(self):
        if False:
            i = 10
            return i + 15
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        mat1 = torch.rand(4, 8, 4, device=self.device_type, requires_grad=True)
        mat2 = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        local_result = torch.bmm(mat1, mat2)
        grad_local_res = torch.ones_like(local_result)
        local_result.backward(grad_local_res)

        def test_placement_comb(placements1: List[Placement], placements2: List[Placement]) -> None:
            if False:
                print('Hello World!')
            mat1_dt = distribute_tensor(mat1, device_mesh, placements1)
            mat2_dt = distribute_tensor(mat2, device_mesh, placements2)
            dist_res = cast(DTensor, torch.bmm(mat1_dt, mat2_dt)).redistribute(device_mesh, [Replicate()])
            dist_local_res = dist_res.to_local()
            self.assertEqual(dist_local_res, local_result)
            grad_dist_res = torch.ones_like(dist_res)
            dist_res.backward(grad_dist_res)
            self.assertIsNotNone(mat1_dt.grad)
            mat1_dt_grad = cast(DTensor, mat1_dt.grad)
            mat1_grad_local = mat1_dt_grad.redistribute(device_mesh, [Replicate()]).to_local()
            self.assertEqual(mat1_grad_local, mat1.grad)
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        shard2_spec = Shard(2)
        replica_spec = Replicate()
        placement_specs = [shard0_spec, shard1_spec, shard2_spec, replica_spec]
        shard_specs_comb = list(itertools.product(placement_specs, placement_specs))
        for spec in shard_specs_comb:
            test_placement_comb([spec[0]], [spec[1]])
if __name__ == '__main__':
    run_tests()