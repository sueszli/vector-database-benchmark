"""A manual controller for model search."""
from typing import Iterator, Sequence
from adanet.experimental.controllers.controller import Controller
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.phases.phase import Phase
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow.compat.v2 as tf

class SequentialController(Controller):
    """A controller where the user specifies the sequences of phase to execute."""

    def __init__(self, phases: Sequence[Phase]):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a SequentialController.\n\n    Args:\n      phases: A list of `Phase` instances.\n    '
        self._phases = phases

    def work_units(self) -> Iterator[WorkUnit]:
        if False:
            i = 10
            return i + 15
        previous_phase = None
        for phase in self._phases:
            for work_unit in phase.work_units(previous_phase):
                yield work_unit
            previous_phase = phase

    def get_best_models(self, num_models: int) -> Sequence[tf.keras.Model]:
        if False:
            for i in range(10):
                print('nop')
        final_phase = self._phases[-1]
        if isinstance(final_phase, ModelProvider):
            return self._phases[-1].get_best_models(num_models)
        raise RuntimeError('Final phase does not provide models.')