"""A phase that automatically ensembles models."""
import abc
import random
from typing import Iterable, Iterator, List
from adanet.experimental.keras.ensemble_model import EnsembleModel
from adanet.experimental.keras.ensemble_model import MeanEnsemble
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.storages.in_memory_storage import InMemoryStorage
from adanet.experimental.storages.storage import Storage
from adanet.experimental.work_units.keras_trainer_work_unit import KerasTrainerWorkUnit
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow.compat.v2 as tf

class EnsembleStrategy(abc.ABC):
    """An abstract ensemble strategy."""

    @abc.abstractmethod
    def __call__(self, candidates: List[tf.keras.Model]) -> Iterable[List[tf.keras.Model]]:
        if False:
            return 10
        pass

class Ensembler(abc.ABC):
    """An abstract ensembler."""

    def __init__(self, loss, optimizer, metrics):
        if False:
            return 10
        self._loss = loss
        self._optimizer = optimizer
        self._metrics = metrics

    @abc.abstractmethod
    def __call__(self, submodels: List[tf.keras.Model]) -> EnsembleModel:
        if False:
            print('Hello World!')
        pass

class MeanEnsembler(Ensembler):
    """An ensembler that averages the weights of submodel outputs."""

    def __init__(self, loss, optimizer, metrics, freeze_submodels=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(loss, optimizer, metrics)
        self._freeze_submodels = freeze_submodels

    def __call__(self, submodels: List[tf.keras.Model]) -> EnsembleModel:
        if False:
            return 10
        ensemble = MeanEnsemble(submodels, freeze_submodels=self._freeze_submodels)
        if self._freeze_submodels:
            for layer in ensemble.layers:
                layer.trainable = False
        ensemble.compile(loss=self._loss, optimizer=tf.keras.optimizers.SGD(0), metrics=self._metrics)
        return ensemble

class GrowStrategy(EnsembleStrategy):
    """An ensemble strategy that adds one candidate to the ensemble at a time."""

    def __call__(self, candidates: List[tf.keras.Model]) -> Iterable[List[tf.keras.Model]]:
        if False:
            for i in range(10):
                print('nop')
        return [[candidate] for candidate in candidates]

class AllStrategy(EnsembleStrategy):
    """An ensemble strategy that adds all candidates to the ensemble."""

    def __call__(self, candidates: List[tf.keras.Model]) -> Iterable[List[tf.keras.Model]]:
        if False:
            while True:
                i = 10
        return [candidates]

class RandomKStrategy(EnsembleStrategy):
    """An ensemble strategy that adds k random candidates (with replacement)."""

    def __init__(self, k, seed=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a RandomKStrategy ensemble strategy.\n\n    Args:\n      k: Number of candidates to sample.\n      seed: Random seed.\n    '
        self._k = k
        self._seed = seed

    def __call__(self, candidates: List[tf.keras.Model]) -> Iterable[List[tf.keras.Model]]:
        if False:
            return 10
        if self._seed:
            random_state = random.getstate()
            random.seed(self._seed)
            candidates = [random.choices(candidates, k=self._k)]
            random_state = random.setstate(random_state)
        else:
            candidates = [random.choices(candidates, k=self._k)]
        return [candidates]

class AutoEnsemblePhase(DatasetProvider, ModelProvider):
    """A phase that automatically ensembles models from a prior phase."""

    def __init__(self, ensemblers: List[Ensembler], ensemble_strategies: List[EnsembleStrategy], storage: Storage=InMemoryStorage(), num_candidates: int=None):
        if False:
            i = 10
            return i + 15
        'Initializes an AutoEnsemblePhase.\n\n    Args:\n      ensemblers: A list of `Ensembler` instances to determine how to combine\n        subnetworks.\n      ensemble_strategies: A list of `EnsembleStrategy` instances to determine\n        which subnetworks compose an ensemble.\n      storage: A `Storage` instance to store models and model metadata.\n      num_candidates: The number of subnetwork candidates to consider from the\n        previous phase. If `None` then all of the subnetworks generated in the\n        previous phase will be considered.\n    '
        super().__init__(storage)
        self._ensemblers = ensemblers
        self._ensemble_strategies = ensemble_strategies
        self._num_candidates = num_candidates

    def work_units(self, previous_phase) -> Iterator[WorkUnit]:
        if False:
            for i in range(10):
                print('nop')
        self._train_dataset = previous_phase.get_train_dataset()
        self._eval_dataset = previous_phase.get_eval_dataset()
        if self._num_candidates:
            candidates = previous_phase.get_best_models(num_models=self._num_candidates)
        else:
            candidates = previous_phase.get_models()
        if self.get_best_models():
            current_best_ensemble = list(self.get_best_models())[0]
        else:
            current_best_ensemble = None
        for ensemble_strategy in self._ensemble_strategies:
            for submodels in ensemble_strategy(candidates):
                for ensembler in self._ensemblers:
                    if current_best_ensemble:
                        previous_ensemble = current_best_ensemble.submodels
                    else:
                        previous_ensemble = []
                    ensemble = ensembler(previous_ensemble + submodels)
                    yield KerasTrainerWorkUnit(ensemble, previous_phase.get_train_dataset(), previous_phase.get_eval_dataset(), self._storage)

    def get_models(self) -> Iterable[tf.keras.Model]:
        if False:
            while True:
                i = 10
        return self._storage.get_models()

    def get_best_models(self, num_models=1) -> Iterable[tf.keras.Model]:
        if False:
            i = 10
            return i + 15
        return self._storage.get_best_models(num_models)

    def get_train_dataset(self) -> tf.data.Dataset:
        if False:
            while True:
                i = 10
        return self._train_dataset

    def get_eval_dataset(self) -> tf.data.Dataset:
        if False:
            i = 10
            return i + 15
        return self._eval_dataset