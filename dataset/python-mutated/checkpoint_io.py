from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from lightning.fabric.utilities.types import _PATH

class CheckpointIO(ABC):
    """Interface to save/load checkpoints as they are saved through the ``Strategy``.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Typically most plugins either use the Torch based IO Plugin; ``TorchCheckpointIO`` but may
    require particular handling depending on the plugin.

    In addition, you can pass a custom ``CheckpointIO`` by extending this class and passing it
    to the Trainer, i.e ``Trainer(plugins=[MyCustomCheckpointIO()])``.

    .. note::

        For some plugins, it is not possible to use a custom checkpoint plugin as checkpointing logic is not
        modifiable.

    """

    @abstractmethod
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Save model/training states as a checkpoint file through state-dump and file-write.\n\n        Args:\n            checkpoint: dict containing model and trainer state\n            path: write-target path\n            storage_options: Optional parameters when saving the model/training states.\n\n        '

    @abstractmethod
    def load_checkpoint(self, path: _PATH, map_location: Optional[Any]=None) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Load checkpoint from a path when resuming or loading ckpt for test/validate/predict stages.\n\n        Args:\n            path: Path to checkpoint\n            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage\n                locations.\n\n        Returns: The loaded checkpoint.\n\n        '

    @abstractmethod
    def remove_checkpoint(self, path: _PATH) -> None:
        if False:
            print('Hello World!')
        'Remove checkpoint file from the filesystem.\n\n        Args:\n            path: Path to checkpoint\n\n        '

    def teardown(self) -> None:
        if False:
            return 10
        'This method is called to teardown the process.'