from copy import deepcopy
import pytest
import torch
from lightning.fabric import Fabric
from tests_fabric.helpers.runif import RunIf

@pytest.mark.parametrize('accelerator', ['cpu', pytest.param('cuda', marks=RunIf(min_cuda_gpus=2))])
def test_ddp_save_load(accelerator, tmp_path):
    if False:
        print('Hello World!')
    'Test that DDP model checkpoints can be saved and loaded successfully.'
    fabric = Fabric(devices=2, accelerator=accelerator, strategy='ddp_spawn')
    fabric.launch(_run_ddp_save_load, tmp_path)

def _run_ddp_save_load(fabric, tmp_path):
    if False:
        return 10
    fabric.seed_everything(0)
    tmp_path = fabric.broadcast(tmp_path)
    model = torch.nn.Linear(2, 2)
    params_before = deepcopy(list(model.parameters()))
    fabric.save(tmp_path / 'saved_before_setup.ckpt', {'model': model})
    wrapped_model = fabric.setup(model)
    fabric.save(tmp_path / 'saved_after_setup.ckpt', {'model': wrapped_model})

    def assert_params_equal(params0, params1):
        if False:
            while True:
                i = 10
        assert all((torch.equal(p0, p1.to(p0.device)) for (p0, p1) in zip(params0, params1)))
    model = torch.nn.Linear(2, 2)
    fabric.load(tmp_path / 'saved_before_setup.ckpt', {'model': model})
    assert_params_equal(params_before, model.parameters())
    fabric.load(tmp_path / 'saved_after_setup.ckpt', {'model': model})
    assert_params_equal(params_before, model.parameters())
    wrapped_model = fabric.setup(model)
    fabric.load(tmp_path / 'saved_before_setup.ckpt', {'model': wrapped_model})
    assert_params_equal(params_before, wrapped_model.parameters())
    fabric.load(tmp_path / 'saved_after_setup.ckpt', {'model': wrapped_model})
    assert_params_equal(params_before, wrapped_model.parameters())