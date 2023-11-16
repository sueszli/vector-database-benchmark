"""A phase that repeats its inner phases."""
from typing import Callable, Iterable, Iterator, List
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.phases.phase import Phase
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow.compat.v2 as tf

class RepeatPhase(DatasetProvider, ModelProvider):
    """A phase that repeats its inner phases."""

    def __init__(self, phase_factory: List[Callable[..., Phase]], repetitions: int):
        if False:
            return 10
        self._phase_factory = phase_factory
        self._repetitions = repetitions
        self._final_phase = None
        'Initializes a RepeatPhase.\n\n    Args:\n      phase_factory: A list of callables that return `Phase` instances.\n      repetitions: Number of times to repeat the phases in the phase factory.\n    '

    def work_units(self, previous_phase: DatasetProvider) -> Iterator[WorkUnit]:
        if False:
            print('Hello World!')
        for _ in range(self._repetitions):
            prev_phase = previous_phase
            for phase in self._phase_factory:
                phase = phase()
                for work_unit in phase.work_units(prev_phase):
                    yield work_unit
                prev_phase = phase
        self._final_phase = prev_phase

    def get_train_dataset(self) -> tf.data.Dataset:
        if False:
            return 10
        if not isinstance(self._final_phase, DatasetProvider):
            raise NotImplementedError('The last phase in repetition does not provide datasets.')
        return self._final_phase.get_train_dataset()

    def get_eval_dataset(self) -> tf.data.Dataset:
        if False:
            print('Hello World!')
        if not isinstance(self._final_phase, DatasetProvider):
            raise NotImplementedError('The last phase in repetition does not provide datasets.')
        return self._final_phase.get_eval_dataset()

    def get_models(self) -> Iterable[tf.keras.Model]:
        if False:
            i = 10
            return i + 15
        if not isinstance(self._final_phase, ModelProvider):
            raise NotImplementedError('The last phase in repetition does not provide models.')
        return self._final_phase.get_models()

    def get_best_models(self, num_models=1) -> Iterable[tf.keras.Model]:
        if False:
            print('Hello World!')
        if not isinstance(self._final_phase, ModelProvider):
            raise NotImplementedError('The last phase in repetition does not provide models.')
        return self._final_phase.get_best_models(num_models)