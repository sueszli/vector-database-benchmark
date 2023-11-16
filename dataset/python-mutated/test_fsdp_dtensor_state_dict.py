import io
from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, skip_if_lt_x_gpu, with_comms

class TestDummyModel(torch.nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        if False:
            return 10
        return torch.rand(8, 8, device='cuda')

class TestDummyModelUneven(torch.nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(5, 10), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(10, 15), nn.ReLU())
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(30, 5))

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        if False:
            while True:
                i = 10
        return torch.rand(5, 5, device='cuda')

class TestFSDPWithDeviceMeshAndDTensor(DTensorTestBase):

    def _create_model(self, is_even_sharded_model, device_mesh=None):
        if False:
            while True:
                i = 10
        dummy_model = TestDummyModel() if is_even_sharded_model else TestDummyModelUneven()
        model = FSDP(dummy_model.cuda(), device_mesh=device_mesh)
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(model.get_input()).sum().backward()
        optim.step()
        return (model, optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize('is_even_sharded_model', [True, False])
    def test_fsdp_init_with_device_mesh(self, is_even_sharded_model):
        if False:
            print('Hello World!')
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        (model, optim) = self._create_model(is_even_sharded_model, device_mesh)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT)
        state_dict = model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(model, optim)
        for v in state_dict.values():
            self.assertEqual(type(v), DTensor)
            self.assertEqual(len(v.placements), 1)
            self.assertEqual(v.placements[0], Shard(dim=0))
            self.assertEqual(v.device_mesh, device_mesh)
        for state in optim_state_dict['state'].values():
            for (k, v) in state.items():
                if k != 'step':
                    self.assertEqual(type(v), DTensor)
                    self.assertEqual(len(v.placements), 1)
                    self.assertEqual(v.placements[0], Shard(dim=0))
                    self.assertEqual(v.device_mesh, device_mesh)
        state_dict_type = FSDP.get_state_dict_type(model)
        self.assertEqual(state_dict_type.state_dict_config._use_dtensor, True)
        self.assertEqual(state_dict_type.optim_state_dict_config._use_dtensor, True)

    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize('offload_to_cpu', [True, False])
    @parametrize('is_even_sharded_model', [True, False])
    def test_dtensor_sharded_tensor_state_dict_identical(self, offload_to_cpu, is_even_sharded_model):
        if False:
            print('Hello World!')
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        (model, optim) = self._create_model(is_even_sharded_model, device_mesh)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu), optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=offload_to_cpu))
        dtensor_sd = model.state_dict()
        dtensor_osd = FSDP.optim_state_dict(model, optim)
        (ref_model, ref_optim) = self._create_model(is_even_sharded_model)
        FSDP.set_state_dict_type(ref_model, StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu), optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=offload_to_cpu))
        sharded_tensor_sd = ref_model.state_dict()
        sharded_tensor_osd = FSDP.optim_state_dict(ref_model, ref_optim)
        for (dtensor_sd_item, sharded_tensor_sd_item) in zip(dtensor_sd.items(), sharded_tensor_sd.items()):
            (k1, v1) = dtensor_sd_item
            (k2, v2) = sharded_tensor_sd_item
            self.assertEqual(k1, k2)
            if len(v2.local_shards()) == 0:
                self.assertEqual(v1.to_local().numel(), 0)
            else:
                self.assertEqual(type(v1), DTensor)
                self.assertEqual(type(v2), ShardedTensor)
                self.assertEqual(v1.to_local(), v2.local_tensor())
                self.assertEqual(v1.to_local().device, v2.local_tensor().device)
        for (dtensor_osd_state, sharded_tensor_osd_state) in zip(dtensor_osd['state'].items(), sharded_tensor_osd['state'].items()):
            self.assertEqual(dtensor_osd_state[0], sharded_tensor_osd_state[0])
            for (dtensor_hyper_param, sharded_tensor_hyper_param) in zip(dtensor_osd_state[1].items(), sharded_tensor_osd_state[1].items()):
                (k1, v1) = dtensor_hyper_param
                (k2, v2) = sharded_tensor_hyper_param
                self.assertEqual(k1, k2)
                if k1 != 'step':
                    if len(v2.local_shards()) == 0:
                        self.assertEqual(v1.to_local().numel(), 0)
                    else:
                        self.assertEqual(type(v1), DTensor)
                        self.assertEqual(type(v2), ShardedTensor)
                        self.assertEqual(v1.to_local(), v2.local_tensor())
                        self.assertEqual(v1.to_local().device, v2.local_tensor().device)
                else:
                    self.assertEqual(v1, v2)

    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize('offload_to_cpu', [True, False])
    @parametrize('is_even_sharded_model', [True, False])
    def test_dtensor_sharded_optim_load_state_dict(self, offload_to_cpu, is_even_sharded_model):
        if False:
            for i in range(10):
                print('nop')
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        (model, optim) = self._create_model(is_even_sharded_model, device_mesh)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT, optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=offload_to_cpu))
        checkpoint = io.BytesIO()
        torch.save(FSDP.optim_state_dict(model, optim), checkpoint)
        ref_optim_state_dict = deepcopy(FSDP.optim_state_dict(model, optim))
        model(model.get_input()).sum().backward()
        optim.step()
        checkpoint.seek(0)
        load_ref_optim_state_dict = torch.load(checkpoint)
        optim.load_state_dict(FSDP.optim_state_dict_to_load(model, optim, load_ref_optim_state_dict))
        new_optim_state_dict = FSDP.optim_state_dict(model, optim)
        for (new_optim_state_dict_item, ref_optim_state_dict_item) in zip(new_optim_state_dict['state'].items(), ref_optim_state_dict['state'].items()):
            self.assertEqual(new_optim_state_dict_item[0], ref_optim_state_dict_item[0])
            for (new_optim_hyper_param, ref_optim_hyper_param) in zip(new_optim_state_dict_item[1].items(), ref_optim_state_dict_item[1].items()):
                (k1, v1) = new_optim_hyper_param
                (k2, v2) = ref_optim_hyper_param
                self.assertEqual(k1, k2)
                self.assertEqual(v1, v2)
                if k1 != 'step':
                    self.assertEqual(type(v1), DTensor)
                    self.assertEqual(type(v2), DTensor)

    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize('offload_to_cpu', [True, False])
    @parametrize('is_even_sharded_model', [True, False])
    def test_dtensor_sharded_model_load_state_dict(self, offload_to_cpu, is_even_sharded_model):
        if False:
            i = 10
            return i + 15
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        (model, optim) = self._create_model(is_even_sharded_model, device_mesh)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu))
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        ref_state_dict = deepcopy(model.state_dict())
        model(model.get_input()).sum().backward()
        optim.step()
        checkpoint.seek(0)
        load_ref_state_dict = torch.load(checkpoint)
        model.load_state_dict(load_ref_state_dict)
        new_state_dict = model.state_dict()
        for ((k1, v1), (k2, v2)) in zip(ref_state_dict.items(), new_state_dict.items()):
            self.assertEqual(k1, k2)
            self.assertEqual(type(v1), DTensor)
            self.assertEqual(type(v2), DTensor)
            self.assertEqual(v1, v2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_raises_warning_or_errors(self):
        if False:
            print('Hello World!')
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        (model, optim) = self._create_model(is_even_sharded_model=True, device_mesh=device_mesh)
        model(model.get_input()).sum().backward()
        optim.step()
        with self.assertRaisesRegex(RuntimeError, 'DeviceMesh is not compatible with LOCAL_STATE_DICT.'):
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                state_dict = model.state_dict()
        with self.assertRaisesRegex(RuntimeError, 'DeviceMesh is not compatible with LOCAL_STATE_DICT.'):
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                optim_state_dict = FSDP.optim_state_dict(model, optim)
        with self.assertLogs('torch.distributed.fsdp._state_dict_utils', level='WARNING') as log:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                state_dict = model.state_dict()
                self.assertEqual(len(log.records), 1)
                self.assertEqual(len(log.output), 1)
                self.assertIn('Found both state_dict_type FULL_STATE_DICT and device_mesh.', log.output[0])
        with self.assertLogs('torch.distributed.fsdp._optim_utils', level='WARNING') as log:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                state_dict = FSDP.optim_state_dict(model, optim)
                self.assertEqual(len(log.records), 1)
                self.assertEqual(len(log.output), 1)
                self.assertIn('Found both state_dict_type FULL_STATE_DICT and device_mesh.', log.output[0])
instantiate_parametrized_tests(TestFSDPWithDeviceMeshAndDTensor)
if __name__ == '__main__':
    run_tests()