import glob
import os
import sys
from unittest.mock import patch
import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch import Callback, Trainer
from tests_pytorch import _PATH_LEGACY
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel
from tests_pytorch.helpers.threading import ThreadExceptionHandler
LEGACY_CHECKPOINTS_PATH = os.path.join(_PATH_LEGACY, 'checkpoints')
CHECKPOINT_EXTENSION = '.ckpt'
with open(os.path.join(_PATH_LEGACY, 'back-compatible-versions.txt')) as fp:
    LEGACY_BACK_COMPATIBLE_PL_VERSIONS = [ln.strip() for ln in fp.readlines()]
LEGACY_BACK_COMPATIBLE_PL_VERSIONS += ['local']

@pytest.mark.parametrize('pl_version', LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@RunIf(sklearn=True)
def test_load_legacy_checkpoints(tmpdir, pl_version: str):
    if False:
        print('Hello World!')
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    with patch('sys.path', [PATH_LEGACY] + sys.path):
        path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f'*{CHECKPOINT_EXTENSION}')))
        assert path_ckpts, f'No checkpoints found in folder "{PATH_LEGACY}"'
        path_ckpt = path_ckpts[-1]
        model = ClassificationModel.load_from_checkpoint(path_ckpt, num_features=24)
        trainer = Trainer(default_root_dir=str(tmpdir))
        dm = ClassifDataModule(num_features=24, length=6000, batch_size=128, n_clusters_per_class=2, n_informative=8)
        res = trainer.test(model, datamodule=dm)
        assert res[0]['test_loss'] <= 0.85, str(res[0]['test_loss'])
        assert res[0]['test_acc'] >= 0.7, str(res[0]['test_acc'])
        print(res)

class LimitNbEpochs(Callback):

    def __init__(self, nb: int):
        if False:
            for i in range(10):
                print('nop')
        self.limit = nb
        self._count = 0

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if False:
            i = 10
            return i + 15
        self._count += 1
        if self._count >= self.limit:
            trainer.should_stop = True

@pytest.mark.parametrize('pl_version', LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@RunIf(sklearn=True)
def test_legacy_ckpt_threading(tmpdir, pl_version: str):
    if False:
        print('Hello World!')
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f'*{CHECKPOINT_EXTENSION}')))
    assert path_ckpts, f'No checkpoints found in folder "{PATH_LEGACY}"'
    path_ckpt = path_ckpts[-1]

    def load_model():
        if False:
            for i in range(10):
                print('nop')
        import torch
        from lightning.pytorch.utilities.migration import pl_legacy_patch
        with pl_legacy_patch():
            _ = torch.load(path_ckpt)
    with patch('sys.path', [PATH_LEGACY] + sys.path):
        t1 = ThreadExceptionHandler(target=load_model)
        t2 = ThreadExceptionHandler(target=load_model)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

@pytest.mark.parametrize('pl_version', LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@RunIf(sklearn=True)
def test_resume_legacy_checkpoints(tmpdir, pl_version: str):
    if False:
        while True:
            i = 10
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    with patch('sys.path', [PATH_LEGACY] + sys.path):
        path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f'*{CHECKPOINT_EXTENSION}')))
        assert path_ckpts, f'No checkpoints found in folder "{PATH_LEGACY}"'
        path_ckpt = path_ckpts[-1]
        dm = ClassifDataModule(num_features=24, length=6000, batch_size=128, n_clusters_per_class=2, n_informative=8)
        model = ClassificationModel(num_features=24)
        stop = LimitNbEpochs(1)
        trainer = Trainer(default_root_dir=str(tmpdir), accelerator='auto', devices=1, precision='16-mixed' if torch.cuda.is_available() else '32-true', callbacks=[stop], max_epochs=21, accumulate_grad_batches=2)
        torch.backends.cudnn.deterministic = True
        trainer.fit(model, datamodule=dm, ckpt_path=path_ckpt)
        res = trainer.test(model, datamodule=dm)
        assert res[0]['test_loss'] <= 0.85, str(res[0]['test_loss'])
        assert res[0]['test_acc'] >= 0.7, str(res[0]['test_acc'])