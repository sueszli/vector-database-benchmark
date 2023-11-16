"""
Deduplicate repeated parameters.

No guarantee for forward-compatibility.
"""
from __future__ import annotations
import logging
import typing
import nni
from .formatting import FormattedParameters, FormattedSearchSpace, ParameterSpec, deformat_parameters
if typing.TYPE_CHECKING:
    from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
_logger = logging.getLogger(__name__)

class Deduplicator:
    """
    A helper for tuners to deduplicate generated parameters.

    When the tuner generates an already existing parameter,
    calling this will return a new parameter generated with grid search.
    Otherwise it returns the orignial parameter object.

    If all parameters have been generated, raise ``NoMoreTrialError``.

    All search space types, including nested choice, are supported.

    Resuming and updating search space are not supported for now.
    It will not raise error, but may return duplicate parameters.

    See random tuner's source code for example usage.
    """

    def __init__(self, formatted_search_space: FormattedSearchSpace):
        if False:
            while True:
                i = 10
        self._space: FormattedSearchSpace = formatted_search_space
        self._never_dup: bool = any((_spec_never_dup(spec) for spec in self._space.values()))
        self._history: set[str] = set()
        self._grid_search: GridSearchTuner | None = None

    def __call__(self, formatted_parameters: FormattedParameters) -> FormattedParameters:
        if False:
            for i in range(10):
                print('nop')
        if self._never_dup or self._not_dup(formatted_parameters):
            return formatted_parameters
        if self._grid_search is None:
            _logger.info(f'Tuning algorithm generated duplicate parameter: {formatted_parameters}')
            _logger.info(f'Use grid search for deduplication.')
            self._init_grid_search()
        while True:
            new = self._grid_search._suggest()
            if new is None:
                raise nni.NoMoreTrialError()
            if self._not_dup(new):
                return new

    def _init_grid_search(self) -> None:
        if False:
            print('Hello World!')
        from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
        self._grid_search = GridSearchTuner()
        self._grid_search.history = self._history
        self._grid_search.space = self._space
        self._grid_search._init_grid()

    def _not_dup(self, formatted_parameters: FormattedParameters) -> bool:
        if False:
            i = 10
            return i + 15
        params = deformat_parameters(formatted_parameters, self._space)
        params_str = typing.cast(str, nni.dump(params, sort_keys=True))
        if params_str in self._history:
            return False
        else:
            self._history.add(params_str)
            return True

    def add_history(self, formatted_parameters: FormattedParameters) -> None:
        if False:
            i = 10
            return i + 15
        params = deformat_parameters(formatted_parameters, self._space)
        params_str = typing.cast(str, nni.dump(params, sort_keys=True))
        if params_str not in self._history:
            self._history.add(params_str)

def _spec_never_dup(spec: ParameterSpec) -> bool:
    if False:
        i = 10
        return i + 15
    if spec.is_nested():
        return False
    if spec.categorical or spec.q is not None:
        return False
    if spec.normal_distributed:
        return spec.sigma > 0
    else:
        return spec.low < spec.high