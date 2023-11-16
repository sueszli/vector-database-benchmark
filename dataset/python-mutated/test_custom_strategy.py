import os
from typing import Any, Mapping
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import SingleDeviceStrategy

@pytest.mark.parametrize('restore_optimizer_and_schedulers', [True, False])
def test_strategy_lightning_restore_optimizer_and_schedulers(tmpdir, restore_optimizer_and_schedulers):
    if False:
        for i in range(10):
            print('nop')

    class TestStrategy(SingleDeviceStrategy):
        load_optimizer_state_dict_called = False

        @property
        def lightning_restore_optimizer(self) -> bool:
            if False:
                while True:
                    i = 10
            return restore_optimizer_and_schedulers

        def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
            if False:
                return 10
            self.load_optimizer_state_dict_called = True
    checkpoint_path = os.path.join(tmpdir, 'model.ckpt')
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    trainer.save_checkpoint(checkpoint_path)
    model = BoringModel()
    strategy = TestStrategy(torch.device('cpu'))
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy=strategy, accelerator='cpu')
    trainer.fit(model, ckpt_path=checkpoint_path)
    assert strategy.load_optimizer_state_dict_called == restore_optimizer_and_schedulers