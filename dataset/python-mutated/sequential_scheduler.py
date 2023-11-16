from typing import Sequence
from snorkel.classification.data import DictDataLoader
from .scheduler import BatchIterator, Scheduler

class SequentialScheduler(Scheduler):
    """Return batches from all dataloaders in sequential order."""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def get_batches(self, dataloaders: Sequence[DictDataLoader]) -> BatchIterator:
        if False:
            print('Hello World!')
        'Return batches from dataloaders sequentially in the order they were given.\n\n        Parameters\n        ----------\n        dataloaders\n            A sequence of dataloaders to get batches from\n\n        Yields\n        ------\n        (batch, dataloader)\n            batch is a tuple of (X_dict, Y_dict) and dataloader is the dataloader\n            that that batch came from. That dataloader will not be accessed by the\n            model; it is passed primarily so that the model can pull the necessary\n            metadata to know what to do with the batch it has been given.\n        '
        for dataloader in dataloaders:
            for batch in dataloader:
                yield (batch, dataloader)