from typing import Dict
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.serve.servable_module_validator import ServableModule, ServableModuleValidator
from torch import Tensor

class ServableBoringModel(BoringModel, ServableModule):

    def configure_payload(self):
        if False:
            i = 10
            return i + 15
        return {'body': {'x': list(range(32))}}

    def configure_serialization(self):
        if False:
            i = 10
            return i + 15

        def deserialize(x):
            if False:
                print('Hello World!')
            return torch.tensor(x, dtype=torch.float)

        def serialize(x):
            if False:
                return 10
            return x.tolist()
        return ({'x': deserialize}, {'output': serialize})

    def serve_step(self, x: Tensor) -> Dict[str, Tensor]:
        if False:
            return 10
        assert torch.equal(x, torch.arange(32, dtype=torch.float))
        return {'output': torch.tensor([0, 1])}

    def configure_response(self):
        if False:
            while True:
                i = 10
        return {'output': [0, 1]}

@pytest.mark.xfail(strict=False, reason='test is too flaky in CI')
def test_servable_module_validator():
    if False:
        for i in range(10):
            print('nop')
    model = ServableBoringModel()
    callback = ServableModuleValidator()
    callback.on_train_start(Trainer(accelerator='cpu'), model)

@pytest.mark.flaky(reruns=3)
def test_servable_module_validator_with_trainer(tmpdir):
    if False:
        print('Hello World!')
    callback = ServableModuleValidator()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=2, limit_val_batches=0, callbacks=[callback], strategy='ddp_spawn', devices=2)
    trainer.fit(ServableBoringModel())