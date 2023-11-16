"""A phase in the AdaNet workflow."""
import sys
from typing import Callable, Iterable, Iterator, Union
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.work_units.keras_tuner_work_unit import KerasTunerWorkUnit
from adanet.experimental.work_units.work_unit import WorkUnit
from kerastuner.engine.tuner import Tuner
import tensorflow.compat.v2 as tf

class KerasTunerPhase(DatasetProvider, ModelProvider):
    """Tunes Keras Model hyperparameters using the Keras Tuner."""

    def __init__(self, tuner: Union[Callable[..., Tuner], Tuner], *search_args, **search_kwargs):
        if False:
            return 10
        'Initializes a KerasTunerPhase.\n\n    Args:\n      tuner: A `kerastuner.tuners.tuner.Tuner` instance or a callable that\n        returns a `kerastuner.tuners.tuner.Tuner` instance.\n      *search_args: Arguments to pass to the tuner search method.\n      **search_kwargs: Keyword arguments to pass to the tuner search method.\n    '
        if callable(tuner):
            self._tuner = tuner()
        else:
            self._tuner = tuner
        self._search_args = search_args
        self._search_kwargs = search_kwargs

    def work_units(self, previous_phase: DatasetProvider) -> Iterator[WorkUnit]:
        if False:
            while True:
                i = 10
        self._train_dataset = previous_phase.get_train_dataset()
        self._eval_dataset = previous_phase.get_eval_dataset()
        yield KerasTunerWorkUnit(self._tuner, *self._search_args, x=self._train_dataset, validation_data=self._eval_dataset, **self._search_kwargs)

    def get_models(self) -> Iterable[tf.keras.Model]:
        if False:
            while True:
                i = 10
        return self._tuner.get_best_models(num_models=sys.maxsize)

    def get_best_models(self, num_models) -> Iterable[tf.keras.Model]:
        if False:
            return 10
        return self._tuner.get_best_models(num_models=num_models)

    def get_train_dataset(self) -> tf.data.Dataset:
        if False:
            for i in range(10):
                print('nop')
        return self._train_dataset

    def get_eval_dataset(self) -> tf.data.Dataset:
        if False:
            return 10
        return self._eval_dataset