"""A phase that provides datasets."""
from typing import Optional
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import Phase
import tensorflow.compat.v2 as tf

class InputPhase(DatasetProvider):
    """A phase that simply relays train and eval datasets."""

    def __init__(self, train_dataset: tf.data.Dataset, eval_dataset: tf.data.Dataset):
        if False:
            for i in range(10):
                print('nop')
        'Initializes an InputPhase.\n\n    Args:\n      train_dataset: A `tf.data.Dataset` for training.\n      eval_dataset: A `tf.data.Dataset` for evaluation.\n    '
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset

    def get_train_dataset(self) -> tf.data.Dataset:
        if False:
            print('Hello World!')
        return self._train_dataset

    def get_eval_dataset(self) -> tf.data.Dataset:
        if False:
            print('Hello World!')
        return self._eval_dataset

    def work_units(self, previous_phase: Optional[Phase]):
        if False:
            i = 10
            return i + 15
        return []