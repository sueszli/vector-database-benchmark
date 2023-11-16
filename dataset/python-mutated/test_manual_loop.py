import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops.optimization.manual import ManualResult
from lightning.pytorch.utilities.exceptions import MisconfigurationException

def test_manual_result():
    if False:
        i = 10
        return i + 15
    training_step_output = {'loss': torch.tensor(25.0, requires_grad=True), 'something': 'jiraffe'}
    result = ManualResult.from_training_step_output(training_step_output)
    asdict = result.asdict()
    assert not asdict['loss'].requires_grad
    assert asdict['loss'] == 25
    assert result.extra == asdict

def test_warning_invalid_trainstep_output(tmpdir):
    if False:
        print('Hello World!')

    class InvalidTrainStepModel(BoringModel):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            return 5
    model = InvalidTrainStepModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match='return a Tensor or have no return'):
        trainer.fit(model)