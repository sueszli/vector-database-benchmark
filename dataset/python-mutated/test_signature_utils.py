import torch
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature

def test_param_in_hook_signature():
    if False:
        for i in range(10):
            print('nop')

    class LightningModule:

        def validation_step(self, dataloader_iter):
            if False:
                return 10
            ...
    model = LightningModule()
    assert is_param_in_hook_signature(model.validation_step, 'dataloader_iter', explicit=True)

    class LightningModule:

        @torch.no_grad()
        def validation_step(self, dataloader_iter):
            if False:
                while True:
                    i = 10
            ...
    model = LightningModule()
    assert is_param_in_hook_signature(model.validation_step, 'dataloader_iter', explicit=True)

    class LightningModule:

        def validation_step(self, *args):
            if False:
                print('Hello World!')
            ...
    model = LightningModule()
    assert not is_param_in_hook_signature(model.validation_step, 'dataloader_iter', explicit=True)
    assert is_param_in_hook_signature(model.validation_step, 'dataloader_iter', explicit=False)

    class LightningModule:

        def validation_step(self, a, b):
            if False:
                i = 10
                return i + 15
            ...
    model = LightningModule()
    assert not is_param_in_hook_signature(model.validation_step, 'dataloader_iter', min_args=3)
    assert is_param_in_hook_signature(model.validation_step, 'dataloader_iter', min_args=2)