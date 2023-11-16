import torch
import torch.distributed as dist
import torch.nn.functional as F
from numpy.testing import assert_array_equal
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor, init_device_mesh
from torch.distributed._tensor.placement_types import _Partial, Replicate, Shard
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

class DummyMLP(torch.nn.Module):

    def __init__(self, device):
        if False:
            return 10
        super().__init__()
        self.net1 = torch.nn.Linear(5, 1024, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(1024, 4, device=device)

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.net2(F.relu(self.net1(x)))

    def reset_parameters(self, *args, **kwargs):
        if False:
            return 10
        with torch.no_grad():
            self.net1.weight.fill_(0.5)
            self.net2.weight.fill_(1)
            self.net1.bias.fill_(1.5)
            self.net2.bias.fill_(1.2)

class DTensorTest(DTensorTestBase):

    @with_comms
    def test_dtensor_constructor(self):
        if False:
            return 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3, requires_grad=True)
        dist_tensor_shape = torch.Size([self.world_size * 3, 3])
        dist_tensor = DTensor(local_tensor, device_mesh, shard_spec, shape=dist_tensor_shape, dtype=local_tensor.dtype, requires_grad=True, stride=local_tensor.stride())
        self.assertEqual(dist_tensor.size(), torch.Size((self.world_size * 3, 3)))
        with self.assertWarnsRegex(UserWarning, 'To construct'):
            DTensor(local_tensor, device_mesh, shard_spec, shape=dist_tensor_shape, dtype=local_tensor.dtype, requires_grad=False, stride=local_tensor.stride())
        local_tensor = torch.randn(3, 3, requires_grad=False)
        with self.assertWarnsRegex(UserWarning, 'To construct'):
            dist_tensor = DTensor(local_tensor, device_mesh, shard_spec, shape=dist_tensor_shape, dtype=local_tensor.dtype, requires_grad=True, stride=local_tensor.stride())

    @with_comms
    def test_meta_dtensor(self):
        if False:
            while True:
                i = 10
        device_mesh = self.build_device_mesh()
        dist_specs = [[Shard(0)], [Replicate()]]
        meta_tensor = torch.randn(1024, 2048, device='meta')
        for dist_spec in dist_specs:
            meta_dtensor = distribute_tensor(meta_tensor, device_mesh, dist_spec)
            self.assertTrue(meta_dtensor.is_meta)
            meta_dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
            torch.nn.init.constant_(meta_dtensor, 1.2)
            value_tensor = torch.empty_like(meta_dtensor.to_local()).fill_(1.2)
            self.assertFalse(meta_dtensor.is_meta)
            self.assertEqual(meta_dtensor.device.type, self.device_type)
            self.assertEqual(meta_dtensor.to_local(), value_tensor)
            meta_dtensor = DTensor.from_local(meta_tensor, device_mesh, dist_spec)
            meta_dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
            torch.nn.init.constant_(meta_dtensor, 1.5)
            self.assertEqual(meta_dtensor.device.type, self.device_type)
            value_tensor = torch.empty_like(meta_dtensor.to_local()).fill_(1.5)
            self.assertEqual(meta_dtensor.to_local(), value_tensor)

    @with_comms
    def test_modules_w_meta_dtensor(self):
        if False:
            for i in range(10):
                print('nop')
        model = DummyMLP('meta')
        device_mesh = self.build_device_mesh()
        model_tp = parallelize_module(model, device_mesh, PairwiseParallel())
        model_tp.to_empty(device=self.device_type)
        model_tp.reset_parameters()
        optim = torch.optim.SGD(model_tp.parameters(), lr=0.1)
        model_regular = DummyMLP(self.device_type)
        model_regular_tp = parallelize_module(model_regular, device_mesh, PairwiseParallel())
        optim_regular = torch.optim.SGD(model_regular_tp.parameters(), lr=0.1)
        model_regular_tp.reset_parameters()
        torch.manual_seed(0)
        inp = torch.randn(20, 5, device=self.device_type)
        output = model_tp(inp)
        output_regular = model_regular_tp(inp)
        self.assertEqual(output, output_regular)
        output.sum().backward()
        output_regular.sum().backward()
        optim.step()
        optim_regular.step()
        torch.manual_seed(1)
        inp = torch.randn(20, 5, device=self.device_type)
        self.assertEqual(model_tp(inp), model_regular_tp(inp))

    @with_comms
    def test_dtensor_stride(self):
        if False:
            print('Hello World!')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = [Shard(0)]
        local_tensor = torch.randn(4, 8)
        global_shape = torch.Size([self.world_size * 4, 8])
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard0_spec)
        self.assertEqual(dist_tensor.stride(), (8, 1))
        shard1_spec = [Shard(1)]
        local_tensor = torch.randn(8, 4)
        global_shape = torch.Size([8, self.world_size * 4])
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard1_spec)
        self.assertEqual(dist_tensor.stride(), (4 * self.world_size, 1))
        local_tensor = torch.randn(8, 4, 8)
        local_tensor_t = local_tensor.permute(1, 2, 0)
        global_shape = torch.Size([4, self.world_size * 8, 8])
        self.assertEqual(local_tensor_t.stride(), (8, 1, 32))
        dist_tensor = DTensor.from_local(local_tensor_t, device_mesh, shard1_spec)
        global_stride = (8 * self.world_size, 1, 32 * self.world_size)
        self.assertEqual(dist_tensor.stride(), global_stride)

    @with_comms
    def test_from_local(self):
        if False:
            print('Hello World!')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.size(), torch.Size([self.world_size * 3, 3]))
        replica_spec = [Replicate()]
        ddp_tensor = DTensor.from_local(local_tensor, device_mesh, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())
        partial_spec = [_Partial()]
        partial_tensor = DTensor.from_local(local_tensor, device_mesh, partial_spec)
        self.assertEqual(partial_tensor.size(), local_tensor.size())
        local_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        local_tensor_temp = local_tensor_with_grad * 3
        dist_tensor = DTensor.from_local(local_tensor_temp, device_mesh, shard_spec)
        self.assertFalse(dist_tensor.is_leaf)
        output = dist_tensor * 3
        self.assertIsInstance(output, DTensor)
        local_grad = torch.ones(3, 3)
        grad_output = DTensor.from_local(local_grad, device_mesh, shard_spec)
        output.backward(grad_output)
        self.assertIsNotNone(local_tensor_with_grad.grad)
        expected_grad = torch.ones(3, 3) * 9
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    def test_from_local_uneven_sharding(self):
        if False:
            print('Hello World!')
        mesh_shape = (self.world_size,)
        device_mesh = init_device_mesh(self.device_type, mesh_shape)
        uneven_dim0_size = self.world_size + 1
        global_tensor = torch.randn(uneven_dim0_size, 2)
        shard_placement = Shard(0)
        (tensor_list, _) = shard_placement._split_tensor(global_tensor, device_mesh.size(dim=0), with_padding=False, contiguous=True)
        dtensor = DTensor.from_local(tensor_list[self.rank], device_mesh, (Shard(0),), shape=global_tensor.size(), stride=global_tensor.stride())
        self.assertEqual(dtensor.size(), global_tensor.size())
        self.assertEqual(dtensor.stride(), global_tensor.stride())

    @with_comms
    def test_from_local_uneven_sharding_raise_error(self):
        if False:
            for i in range(10):
                print('nop')
        mesh_shape = (self.world_size,)
        device_mesh = init_device_mesh(self.device_type, mesh_shape)
        uneven_dim0_size = self.world_size + 1
        global_tensor = torch.randn(uneven_dim0_size, 2)
        shard_placement = Shard(0)
        (tensor_list, _) = shard_placement._split_tensor(global_tensor, device_mesh.size(dim=0), with_padding=False, contiguous=True)
        with self.assertRaisesRegex(RuntimeError, 'Please pass both shape and stride at the same time.'):
            dtensor = DTensor.from_local(tensor_list[self.rank], device_mesh, (Shard(0),), shape=global_tensor.size())
        with self.assertRaisesRegex(RuntimeError, 'Please pass both shape and stride at the same time.'):
            dtensor = DTensor.from_local(tensor_list[self.rank], device_mesh, (Shard(0),), stride=global_tensor.stride())

    @with_comms
    def test_from_local_negative_dim(self):
        if False:
            for i in range(10):
                print('nop')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(-1)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.placements[0].dim, 1)

    @with_comms
    def test_to_local(self):
        if False:
            return 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = (Shard(0),)
        dist_tensor_shape = torch.Size([self.world_size * 3, 3])
        local_tensor_with_grad = torch.randn(3, 3, device=self.device_type, requires_grad=True)
        sharded_tensor = DTensor(local_tensor_with_grad, device_mesh, shard_spec, shape=dist_tensor_shape, dtype=local_tensor_with_grad.dtype, requires_grad=True, stride=local_tensor_with_grad.stride())
        self.assertEqual(sharded_tensor.size(), dist_tensor_shape)
        self.assertEqual(sharded_tensor.to_local(), local_tensor_with_grad)
        temp_st = sharded_tensor * 3
        new_tensor_with_grad = torch.randn(3, 3, device=self.device_type, requires_grad=True)
        res = temp_st.to_local() + new_tensor_with_grad
        res.sum().backward()
        self.assertIsNotNone(sharded_tensor.grad)
        self.assertEqual(sharded_tensor.grad.to_local(), torch.ones(3, 3) * 3)
        res = sharded_tensor.to_local()
        model = torch.nn.ReLU()
        res.register_hook(lambda grad: grad.t())
        target = torch.randn(3, 3, device=self.device_type)
        mae_loss = torch.nn.L1Loss()
        output = mae_loss(model(res), target)
        try:
            output.backward()
        except RuntimeError:
            self.assertEqual(sharded_tensor.grad.stride(), [1, 3 * self.world_size])

    @with_comms
    def test_to_local_grad_hint(self):
        if False:
            while True:
                i = 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = (Shard(0),)
        global_tensor = torch.ones(8, 3, requires_grad=True)
        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, shard_spec)
        local_out = sharded_dtensor.redistribute(placements=[Replicate()]).to_local(grad_placements=[_Partial()])
        local_out.sum().backward()
        replica_grad = sharded_dtensor.grad.full_tensor()
        self.assertEqual(replica_grad, global_tensor * self.world_size)

    @with_comms
    def test_full_tensor_sync(self):
        if False:
            return 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = (Shard(0),)
        global_tensor = torch.ones(8, 3, requires_grad=True)
        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, shard_spec)
        full_out = sharded_dtensor.full_tensor()
        self.assertFalse(isinstance(full_out, AsyncCollectiveTensor))
        self.assertEqual(full_out, global_tensor)

    @with_comms
    def test_full_tensor_grad_hint(self):
        if False:
            return 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = (Shard(0),)
        global_tensor = torch.ones(8, 3, requires_grad=True)
        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, shard_spec)
        local_out = sharded_dtensor.full_tensor(grad_placements=[_Partial()])
        local_out.sum().backward()
        replica_grad = sharded_dtensor.grad.full_tensor()
        self.assertEqual(replica_grad, global_tensor * self.world_size)

    @with_comms
    def test_dtensor_new_empty_strided(self):
        if False:
            for i in range(10):
                print('nop')
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local_tensor = torch.randn(8, 8, requires_grad=True, device=self.device_type)
        my_dtensor = distribute_tensor(local_tensor, device_mesh, [Shard(0)])
        new_strided_dtensor = my_dtensor.new_empty_strided((8, 8), (8, 1), requires_grad=True)
        self.assertEqual(new_strided_dtensor.shape, my_dtensor.shape)
        new_strided_dtensor.sum().backward()
        self.assertIsNotNone(new_strided_dtensor.grad)
        self.assertIsInstance(new_strided_dtensor.grad, DTensor)
        my_dtensor.to_local().sum().backward()
        local_tensor.sum().backward()
        self.assertEqual(my_dtensor.grad, new_strided_dtensor.grad)
        self.assertEqual(my_dtensor.grad.redistribute(placements=[Replicate()]).to_local(), local_tensor.grad)

    @with_comms
    def test_dtensor_async_output(self):
        if False:
            for i in range(10):
                print('nop')
        from torch.distributed._functional_collectives_impl import _tensor_needs_wait
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(dt):
            if False:
                print('Hello World!')
            dt_out_redistribute = dt.redistribute(mesh, [Replicate()])
            dt_out_redistribute_view = dt_out_redistribute.view(dt_out_redistribute.shape)
            local_tensor = dt_out_redistribute_view.to_local()
            return local_tensor
        x = torch.ones((4, 2), device=self.device_type)
        dt = distribute_tensor(x, mesh, [Shard(0)])
        out = fn(dt)
        self.assertEqual(type(out), AsyncCollectiveTensor)
        self.assertTrue(_tensor_needs_wait(out.elem))
        out_view = out.view(-1)
        self.assertEqual(type(out_view), AsyncCollectiveTensor)
        self.assertTrue(_tensor_needs_wait(out_view.elem))
        ref = torch.ones((4, 2), device=self.device_type) + 1
        ref = ref.view(-1)
        out_data = out_view + 1
        self.assertEqual(type(out_data), torch.Tensor)
        self.assertEqual(out_data, ref)

    @with_comms
    def test_from_local_then_to_local(self):
        if False:
            while True:
                i = 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor_with_grad = torch.randn(3, 3, device=self.device_type, requires_grad=True)
        local_tensor_temp = local_tensor_with_grad + 8
        dist_tensor = DTensor.from_local(local_tensor_temp, device_mesh, shard_spec)
        self.assertFalse(dist_tensor.is_leaf)
        output = dist_tensor * 6
        self.assertIsInstance(output, DTensor)
        new_tensor_with_grad = torch.randn(3, 3, device=self.device_type, requires_grad=True)
        res = output.to_local() + new_tensor_with_grad
        res.sum().backward()
        self.assertIsNotNone(local_tensor_with_grad.grad)
        expected_grad = torch.ones(3, 3) * 6
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    def test_dtensor_spec_read_only_after_set(self):
        if False:
            print('Hello World!')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        shard_spec[0] = Replicate()
        self.assertTrue(sharded_tensor.placements is not shard_spec)
        self.assertNotEqual(sharded_tensor.placements, shard_spec)

    @with_comms
    def test_dtensor_spec_hash(self):
        if False:
            return 10
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        local_tensor2 = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        sharded_tensor2 = DTensor.from_local(local_tensor2, device_mesh, shard_spec)
        self.assertEqual(hash(sharded_tensor._spec), hash(sharded_tensor2._spec))
        local_tensor3 = torch.ones(3, 3)
        replica_spec = [Replicate()]
        replica_tensor = DTensor.from_local(local_tensor3, device_mesh, replica_spec, run_check=False)
        self.assertNotEqual(hash(sharded_tensor._spec), hash(replica_tensor._spec))

    @with_comms
    def test_dtensor_properties(self):
        if False:
            for i in range(10):
                print('nop')
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.device.type, self.device_type)

    @with_comms
    def test_dtensor_save_load(self):
        if False:
            while True:
                i = 10
        import io
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        buffer = io.BytesIO()
        torch.save(sharded_tensor, buffer)
        buffer.seek(0)
        reloaded_st = torch.load(buffer)
        self.assertEqual(sharded_tensor, reloaded_st)

class DTensorMeshTest(DTensorTestBase):

    @property
    def world_size(self):
        if False:
            return 10
        return 8

    def sub_mesh_assert_equal(self, mesh, exp_in_mesh, exp_out_of_mesh, tensor):
        if False:
            for i in range(10):
                print('nop')
        if self.rank in mesh:
            self.assertEqual(tensor, exp_in_mesh)
        else:
            self.assertEqual(tensor, exp_out_of_mesh)

    @with_comms
    def test_dtensor_device_mesh_device_conversion(self):
        if False:
            for i in range(10):
                print('nop')
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_dtensor_api_device_mesh_context_manager(self):
        if False:
            print('Hello World!')
        with DeviceMesh(self.device_type, list(range(self.world_size))) as mesh:
            shard_spec = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(local_tensor, device_mesh=mesh, placements=shard_spec)
        with DeviceMesh(self.device_type, list(range(self.world_size))):
            shard_spec = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(local_tensor, placements=shard_spec)
            replica_spec = [Replicate()]
            replica_tensor = sharded_tensor.redistribute(placements=replica_spec)
            self.assertEqual(replica_tensor.size(), torch.Size([3 * self.world_size, 3]))
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            shard_spec = [Shard(0)]
            global_shape = torch.Size([3 * self.world_size, 3])
            global_tensor = torch.randn(global_shape)
            sharded_tensor = distribute_tensor(global_tensor, placements=shard_spec)
            self.assertEqual(sharded_tensor.to_local().shape, torch.Size([3, 3]))
            mesh_2d = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 4))
            with mesh_2d:
                shard_2d_spec = [Shard(0), Replicate()]
                tensor_2d = distribute_tensor(global_tensor, placements=shard_2d_spec)
                self.assertEqual(tensor_2d.to_local().shape, torch.Size([3 * 4, 3]))
            sharded_after_2d = distribute_tensor(global_tensor, placements=shard_spec)
            self.assertEqual(sharded_after_2d.to_local().shape, torch.Size([3, 3]))

    @with_comms
    def test_dtensor_2d_mesh(self):
        if False:
            return 10
        mesh_tensor = torch.arange(self.world_size).reshape(2, 4)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        shard_spec = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([3 * mesh.size(0), 3 * mesh.size(1)]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)
        shard_same_dim_spec = [Shard(0), Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_same_dim_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([3 * self.world_size, 3]))

    @with_comms
    def test_device_mesh_nd(self):
        if False:
            i = 10
            return i + 15
        mesh_tensor = torch.arange(self.world_size).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        shard_spec = [Shard(0), Shard(1), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)
        shard_spec = [Shard(0), Shard(0), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([12, 3, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_dtensor_spec_local_shard_offset(self):
        if False:
            i = 10
            return i + 15
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 4))
        tensor_shape = (3 * self.world_size, 3 * self.world_size)
        shard_spec_and_offsets = [([Shard(0), Replicate()], (3 * (self.world_size // 2) * (self.rank // 4), 0)), ([Shard(1), Replicate()], (0, 3 * (self.world_size // 2) * (self.rank // 4))), ([Replicate(), Shard(0)], (3 * (self.world_size // 4) * (self.rank % 4), 0)), ([Replicate(), Shard(1)], (0, 3 * (self.world_size // 4) * (self.rank % 4)))]
        from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
        logical_tensor = torch.randn(tensor_shape)
        for (shard_spec, expected_shard_offsets) in shard_spec_and_offsets:
            dtensor = distribute_tensor(logical_tensor, device_mesh, shard_spec)
            (_, offset) = compute_local_shape_and_global_offset(dtensor.shape, device_mesh, dtensor.placements)
            self.assertEqual(expected_shard_offsets, offset)

    @with_comms
    def test_from_local_sub_mesh(self):
        if False:
            return 10
        mesh = DeviceMesh(self.device_type, [0, 2])
        local_tensor = torch.ones(3, 4)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        self.assertEqual(dtensor.size(), torch.Size([6, 4]))
        self.sub_mesh_assert_equal(mesh.mesh, torch.ones(3, 4), torch.tensor([]), dtensor.to_local())
        dtensor = dtensor + 2
        self.sub_mesh_assert_equal(mesh.mesh, torch.ones(3, 4) + 2, torch.tensor([]), dtensor.to_local())

    @with_comms
    def test_default_value_sub_mesh(self):
        if False:
            return 10
        mesh = DeviceMesh(self.device_type, [0, 2])
        local_tensor1 = torch.ones(4, 3)
        local_tensor2 = torch.ones(4, 3)
        dtensor1 = DTensor.from_local(local_tensor1, mesh, [Shard(0)])
        dtensor2 = DTensor.from_local(local_tensor2, mesh, [Shard(0)])
        local_res = dtensor1.equal(dtensor2)
        self.sub_mesh_assert_equal(mesh.mesh, True, True, local_res)
        local_tensor = torch.ones(4, 3)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)]).sum()
        self.sub_mesh_assert_equal(mesh.mesh, torch.tensor(12.0), torch.tensor(0.0), dtensor.to_local())
        local_tensor = torch.ones(3, 4)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        dtensor_list = dtensor.split([2, 2], dim=1)
        self.sub_mesh_assert_equal(mesh.mesh, [torch.ones(3, 2)] * 2, [torch.tensor([])] * 2, [dt.to_local() for dt in dtensor_list])

    @with_comms
    def test_redistribute_sub_mesh(self):
        if False:
            i = 10
            return i + 15
        mesh = DeviceMesh(self.device_type, [0, 2])
        local_tensor1 = torch.ones(4, 3)
        sharded_dtensor = DTensor.from_local(local_tensor1, mesh, [Shard(0)])
        replicated_dtensor = sharded_dtensor.redistribute(placements=[Replicate()])
        self.sub_mesh_assert_equal(mesh.mesh, torch.ones(8, 3), torch.tensor([]), replicated_dtensor.to_local())
        sharded_again = replicated_dtensor.redistribute(placements=[Shard(0)])
        self.sub_mesh_assert_equal(mesh.mesh, torch.ones(4, 3), torch.tensor([]), sharded_again.to_local())

class TestDTensorPlacementTypes(DTensorTestBase):

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return 8

    def _create_tensor(self, size):
        if False:
            for i in range(10):
                print('nop')
        torch.manual_seed(0)
        tensor = torch.rand(size)
        if self.device_type == 'cuda':
            return tensor.cuda()
        else:
            return tensor

    @with_comms
    def test_split_tensor(self) -> None:
        if False:
            i = 10
            return i + 15
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        shard_placement = Shard(0)
        for size in range(8):
            tensor = self._create_tensor(size)
            if size == 0:
                with self.assertRaisesRegex(Exception, 'Tensor size along dim0 is 0. There is nothing to be sharded.'):
                    (_, _) = shard_placement._split_tensor(tensor, mesh.size(), with_padding=True, contiguous=True)
            else:
                (splitted_tensor_list, pad_sizes) = shard_placement._split_tensor(tensor, mesh.size(), with_padding=True, contiguous=True)
                expected_pad_sizes = [0 if idx < size else 1 for (idx, _) in enumerate(range(dist.get_world_size()))]
                assert_array_equal(expected_pad_sizes, pad_sizes)
                unpadded_list = [shard_placement._unpad_tensor(tensor, pad_sizes[i]) if pad_sizes[i] > 0 else tensor for (i, tensor) in enumerate(splitted_tensor_list)]
                expected_is_tensor_empty = [False if idx < size else True for (idx, _) in enumerate(range(dist.get_world_size()))]
                is_tensor_empty = [False if unpadded_tensor.numel() > 0 else True for unpadded_tensor in unpadded_list]
                assert_array_equal(expected_is_tensor_empty, is_tensor_empty)
if __name__ == '__main__':
    run_tests()