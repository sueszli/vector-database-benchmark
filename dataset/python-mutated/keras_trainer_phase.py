"""A phase in the AdaNet workflow."""
from typing import Callable, Iterable, Iterator, Union
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.storages.in_memory_storage import InMemoryStorage
from adanet.experimental.storages.storage import Storage
from adanet.experimental.work_units.keras_trainer_work_unit import KerasTrainerWorkUnit
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow.compat.v2 as tf

class KerasTrainerPhase(DatasetProvider, ModelProvider):
    """Trains Keras models."""

    def __init__(self, models: Union[Iterable[tf.keras.Model], Callable[[], Iterable[tf.keras.Model]]], storage: Storage=InMemoryStorage()):
        if False:
            i = 10
            return i + 15
        'Initializes a KerasTrainerPhase.\n\n    Args:\n      models: A list of `tf.keras.Model` instances or a list of callables that\n        return `tf.keras.Model` instances.\n      storage: A `Storage` instance.\n    '
        super().__init__(storage)
        self._models = models

    def work_units(self, previous_phase: DatasetProvider) -> Iterator[WorkUnit]:
        if False:
            for i in range(10):
                print('nop')
        self._train_dataset = previous_phase.get_train_dataset()
        self._eval_dataset = previous_phase.get_eval_dataset()
        models = self._models
        if callable(models):
            models = models()
        for model in models:
            yield KerasTrainerWorkUnit(model, self._train_dataset, self._eval_dataset, self._storage)

    def get_models(self) -> Iterable[tf.keras.Model]:
        if False:
            for i in range(10):
                print('nop')
        return self._storage.get_models()

    def get_best_models(self, num_models) -> Iterable[tf.keras.Model]:
        if False:
            while True:
                i = 10
        return self._storage.get_best_models(num_models)

    def get_train_dataset(self) -> tf.data.Dataset:
        if False:
            for i in range(10):
                print('nop')
        return self._train_dataset

    def get_eval_dataset(self) -> tf.data.Dataset:
        if False:
            for i in range(10):
                print('nop')
        return self._eval_dataset