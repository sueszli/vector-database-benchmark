import logging
import os
from typing import Any, Callable, Dict, Optional
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.utilities.cloud_io import _atomic_save, get_filesystem
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.types import _PATH
log = logging.getLogger(__name__)

class TorchCheckpointIO(CheckpointIO):
    """CheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save and load checkpoints respectively,
    common for most use cases.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any]=None) -> None:
        if False:
            while True:
                i = 10
        'Save model/training states as a checkpoint file through state-dump and file-write.\n\n        Args:\n            checkpoint: dict containing model and trainer state\n            path: write-target path\n            storage_options: not used in ``TorchCheckpointIO.save_checkpoint``\n\n        Raises:\n            TypeError:\n                If ``storage_options`` arg is passed in\n\n        '
        if storage_options is not None:
            raise TypeError(f"`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO` to define how you'd like to use `storage_options`.")
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_save(checkpoint, path)

    def load_checkpoint(self, path: _PATH, map_location: Optional[Callable]=lambda storage, loc: storage) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Loads checkpoint using :func:`torch.load`, with additional handling for ``fsspec`` remote loading of files.\n\n        Args:\n            path: Path to checkpoint\n            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage\n                locations.\n\n        Returns: The loaded checkpoint.\n\n        Raises:\n            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem\n\n        '
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f'Checkpoint file not found: {path}')
        return pl_load(path, map_location=map_location)

    def remove_checkpoint(self, path: _PATH) -> None:
        if False:
            print('Hello World!')
        'Remove checkpoint file from the filesystem.\n\n        Args:\n            path: Path to checkpoint\n\n        '
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug(f'Removed checkpoint: {path}')