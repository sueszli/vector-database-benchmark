import warnings
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

class TestModel(BoringModel):

    def training_step(self, batch, batch_idx):
        if False:
            return 10
        return self.step(batch[0])

def test_no_depre_without_epoch_end(tmpdir):
    if False:
        print('Hello World!')
    'Tests that only training_step can be used.'
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=2, log_every_n_steps=1, enable_model_summary=False)
    with warnings.catch_warnings(record=True) as w:
        trainer.fit(model)
        for msg in w:
            assert 'should not return anything ' not in str(msg)