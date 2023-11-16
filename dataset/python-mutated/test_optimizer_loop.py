from typing import Dict, Generic, Iterator, Mapping, TypeVar
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops.optimization.automatic import ClosureResult
from lightning.pytorch.utilities.exceptions import MisconfigurationException

def test_closure_result_deepcopy():
    if False:
        while True:
            i = 10
    closure_loss = torch.tensor(123.45)
    result = ClosureResult(closure_loss)
    assert closure_loss.data_ptr() == result.closure_loss.data_ptr()
    assert closure_loss.data_ptr() != result.loss.data_ptr()
    copy = result.asdict()
    assert result.loss == copy['loss']
    assert copy.keys() == {'loss'}
    assert id(result.loss) == id(copy['loss'])
    assert result.loss.data_ptr() == copy['loss'].data_ptr()

def test_closure_result_apply_accumulation():
    if False:
        i = 10
        return i + 15
    closure_loss = torch.tensor(25.0)
    result = ClosureResult.from_training_step_output(closure_loss, 5)
    assert result.loss == 5
T = TypeVar('T')

class OutputMapping(Generic[T], Mapping[str, T]):

    def __init__(self, d: Dict[str, T]) -> None:
        if False:
            print('Hello World!')
        self.d: Dict[str, T] = d

    def __iter__(self) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        return iter(self.d)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.d)

    def __getitem__(self, key: str) -> T:
        if False:
            while True:
                i = 10
        return self.d[key]

@pytest.mark.parametrize('case', [(5.0, 'must return a Tensor, a dict, or None'), ({'a': 5}, "the 'loss' key needs to be present"), (OutputMapping({'a': 5}), "the 'loss' key needs to be present")])
def test_warning_invalid_trainstep_output(tmpdir, case):
    if False:
        return 10
    (output, match) = case

    class InvalidTrainStepModel(BoringModel):

        def training_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            return output
    model = InvalidTrainStepModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match=match):
        trainer.fit(model)