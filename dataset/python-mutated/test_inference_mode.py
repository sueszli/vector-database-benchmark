from unittest import mock
from unittest.mock import Mock
import pytest
import torch
from lightning.fabric.utilities.imports import _TORCH_EQUAL_2_0
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops import _Loop
from lightning.pytorch.loops.utilities import _no_grad_context

@pytest.mark.parametrize('trainer_fn', ['validate', 'test', 'predict'])
def test_eval_inference_mode(tmp_path, trainer_fn):
    if False:
        while True:
            i = 10

    class BoringModelNoGrad(BoringModel):

        def assert_not_enabled(self):
            if False:
                return 10
            assert not torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
        on_test_start = assert_not_enabled
        on_validation_start = assert_not_enabled
        on_predict_start = assert_not_enabled

    class BoringModelForInferenceMode(BoringModel):

        def assert_enabled(self):
            if False:
                print('Hello World!')
            assert not torch.is_grad_enabled()
            assert torch.is_inference_mode_enabled()
        on_test_start = assert_enabled
        on_validation_start = assert_enabled
        on_predict_start = assert_enabled
    trainer = Trainer(default_root_dir=tmp_path, logger=False, inference_mode=False, fast_dev_run=True)
    getattr(trainer, trainer_fn)(BoringModelNoGrad())
    trainer = Trainer(logger=False, inference_mode=True, fast_dev_run=True)
    getattr(trainer, trainer_fn)(BoringModelForInferenceMode())

def test_no_grad_context():
    if False:
        return 10
    trainer = Mock()

    class Foo:

        @_no_grad_context
        def run(self):
            if False:
                return 10
            ...
    f = Foo()
    with pytest.raises(TypeError, match='Foo` needs to be a Loop'):
        f.run()

    class Foo(_Loop):

        @_no_grad_context
        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            ...
    f = Foo(trainer)
    with pytest.raises(TypeError, match='Foo.inference_mode` needs to be defined'):
        f.run()

    class Foo(_Loop):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__(trainer)
            self.inference_mode = False

        @_no_grad_context
        def run(self):
            if False:
                print('Hello World!')
            ...
    f = Foo()
    with mock.patch('torch.no_grad') as no_grad_mock:
        f.run()
    no_grad_mock.assert_called_once_with()
    f.inference_mode = True
    with mock.patch('torch.inference_mode') as inference_mode_mock:
        f.run()
    if not _TORCH_EQUAL_2_0:
        inference_mode_mock.assert_called_once_with()