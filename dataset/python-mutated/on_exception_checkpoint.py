"""
On exception checkpointing
==========================

Automatically save a checkpoints on exception.
"""
import os
from typing import Any
import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import Checkpoint

class OnExceptionCheckpoint(Checkpoint):
    """Used to save a checkpoint on exception.

    Args:
        dirpath: directory to save the checkpoint file.
        filename: checkpoint filename. This must not include the extension.

    Raises:
        ValueError:
            If ``filename`` is empty.


    Example:
        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import OnExceptionCheckpoint
        >>> trainer = Trainer(callbacks=[OnExceptionCheckpoint(".")])

    """
    FILE_EXTENSION = '.ckpt'

    def __init__(self, dirpath: _PATH, filename: str='on_exception') -> None:
        if False:
            print('Hello World!')
        super().__init__()
        if not filename:
            raise ValueError('The filename cannot be empty')
        self.dirpath = dirpath
        self.filename = filename

    @property
    def ckpt_path(self) -> str:
        if False:
            print('Hello World!')
        return os.path.join(self.dirpath, self.filename + self.FILE_EXTENSION)

    def on_exception(self, trainer: 'pl.Trainer', *_: Any, **__: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        trainer.save_checkpoint(self.ckpt_path)

    def teardown(self, trainer: 'pl.Trainer', *_: Any, **__: Any) -> None:
        if False:
            i = 10
            return i + 15
        trainer.strategy.remove_checkpoint(self.ckpt_path)