import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.random as random
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.style import ColwiseParallel, make_input_replicate_1d, make_output_replicate_1d
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, MLPModule, with_comms

class TensorParallelRandomStateTests(DTensorTestBase):

    def get_tensor_slice(self, idx, n, large_tensor):
        if False:
            for i in range(10):
                print('nop')
        shape = large_tensor.shape
        assert shape[0] % n == 0
        local_shape = [shape[0] // n, shape[1]]
        slice_idx = [slice(idx * local_shape[0], (idx + 1) * local_shape[0]), slice(local_shape[1])]
        return large_tensor[slice_idx]

    def check_gathered_tensors(self, self_rank, size, gathered_tensors, assertFunc):
        if False:
            for i in range(10):
                print('nop')
        for other_rank in range(size):
            if self_rank != other_rank:
                assertFunc(self.get_tensor_slice(self_rank, size, gathered_tensors), self.get_tensor_slice(other_rank, size, gathered_tensors))

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_model_init(self):
        if False:
            print('Hello World!')
        mesh = torch.arange(self.world_size).reshape(2, 2)
        device_mesh = DeviceMesh(self.device_type, mesh)
        tp_rank = device_mesh.get_coordinate()[0]
        dp_rank = device_mesh.get_coordinate()[1]
        for enable_distribute_flag in [False, True]:
            model = MLPModule(device='meta')
            model_tp = parallelize_module(model, device_mesh, {'net1': ColwiseParallel(make_input_replicate_1d, make_output_replicate_1d), 'net2': ColwiseParallel(make_input_replicate_1d, make_output_replicate_1d)})
            torch.cuda.manual_seed(dp_rank)
            random._rng_tracker.distribute_region_enabled = enable_distribute_flag
            self.assertTrue(model_tp.net1.weight.is_meta)
            model_tp.to_empty(device=self.device_type)
            model_tp.reset_parameters()
            for dtensor in [model_tp.net1.weight, model_tp.net2.weight]:
                _1d_mesh = dtensor.device_mesh
                assert _1d_mesh.ndim == 1
                tensor_local = dtensor.to_local()
                tensor_gather = funcol.all_gather_tensor(tensor_local, gather_dim=0, group=(_1d_mesh, 0))
                self.assertEqual(_1d_mesh.get_coordinate()[0], tp_rank)

                def tp_weights_assert(tensor1, tensor2):
                    if False:
                        return 10
                    if enable_distribute_flag:
                        self.assertNotEqual(tensor1, tensor2)
                    else:
                        self.assertEqual(tensor1, tensor2)
                self.check_gathered_tensors(tp_rank, 2, tensor_gather, tp_weights_assert)
                tensor_gather = funcol.all_gather_tensor(tensor_local, gather_dim=0, group=(_1d_mesh, 1))

                def dp_weights_assert(tensor1, tensor2):
                    if False:
                        print('Hello World!')
                    if enable_distribute_flag:
                        self.assertEqual(tensor1, tensor2)
                    else:
                        self.assertNotEqual(tensor1, tensor2)
                self.check_gathered_tensors(dp_rank, 2, tensor_gather, dp_weights_assert)
if __name__ == '__main__':
    run_tests()