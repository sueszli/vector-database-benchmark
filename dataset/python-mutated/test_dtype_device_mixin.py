import torch.nn as nn
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf

class SubSubModule(_DeviceDtypeModuleMixin):
    pass

class SubModule(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.module = SubSubModule()

class TopModule(BoringModel):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.module = SubModule()

class DeviceAssertCallback(Callback):

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if False:
            i = 10
            return i + 15
        rank = trainer.local_rank
        assert isinstance(model, TopModule)
        assert model.device.index is None and rank == 0 or model.device.index == rank
        assert model.device == model.module.module.device

@RunIf(min_cuda_gpus=2)
def test_submodules_multi_gpu_ddp_spawn(tmpdir):
    if False:
        i = 10
        return i + 15
    model = TopModule()
    trainer = Trainer(default_root_dir=tmpdir, strategy='ddp_spawn', accelerator='gpu', devices=2, callbacks=[DeviceAssertCallback()], max_steps=1)
    trainer.fit(model)