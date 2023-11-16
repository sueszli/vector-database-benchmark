import pytest
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

@pytest.mark.parametrize('single_cb', [False, True])
def test_train_step_no_return(tmpdir, single_cb: bool):
    if False:
        print('Hello World!')
    'Tests that only training_step can be used.'

    class CB(Callback):

        def on_train_batch_end(self, trainer, pl_module, outputs, *_):
            if False:
                for i in range(10):
                    print('nop')
            assert 'loss' in outputs

        def on_validation_batch_end(self, trainer, pl_module, outputs, *_):
            if False:
                while True:
                    i = 10
            assert 'x' in outputs

        def on_test_batch_end(self, trainer, pl_module, outputs, *_):
            if False:
                print('Hello World!')
            assert 'x' in outputs

    class TestModel(BoringModel):

        def on_train_batch_end(self, outputs, *_):
            if False:
                i = 10
                return i + 15
            assert 'loss' in outputs

        def on_validation_batch_end(self, outputs, *_):
            if False:
                return 10
            assert 'x' in outputs

        def on_test_batch_end(self, outputs, *_):
            if False:
                for i in range(10):
                    print('nop')
            assert 'x' in outputs
    model = TestModel()
    trainer = Trainer(callbacks=CB() if single_cb else [CB()], default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=1, log_every_n_steps=1, enable_model_summary=False)
    assert any((isinstance(c, CB) for c in trainer.callbacks))
    trainer.fit(model)