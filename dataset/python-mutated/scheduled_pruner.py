from __future__ import annotations
from collections import defaultdict
import logging
from typing import Dict, List
import torch
from ..base.compressor import Pruner
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING
from .basic_pruner import LevelPruner, L1NormPruner, L2NormPruner, FPGMPruner
from .slim_pruner import SlimPruner
from .taylor_pruner import TaylorPruner
_logger = logging.getLogger(__name__)

class ScheduledPruner(Pruner):

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None=None, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model, config_list, evaluator, existed_wrappers)
        self.evaluator: Evaluator
        self.sparse_goals: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
        self._goals_initialized = False
        self._scheduled_keys = ['sparse_ratio', 'sparse_threshold', 'max_sparse_ratio', 'min_sparse_ratio']

    def _init_sparse_goals(self):
        if False:
            print('Hello World!')
        if self._goals_initialized:
            _logger.warning('Sparse goals have already initialized.')
            return
        for (module_name, ts) in self._target_spaces.items():
            for (target_name, target_space) in ts.items():
                self.sparse_goals[module_name][target_name] = {}
                for scheduled_key in self._scheduled_keys:
                    if getattr(target_space, scheduled_key) is not None:
                        self.sparse_goals[module_name][target_name][scheduled_key] = getattr(target_space, scheduled_key)
        self._goals_initialized = True

    def update_sparse_goals(self, current_times: int):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def _update_sparse_goals_by_ratio(self, ratio: float):
        if False:
            for i in range(10):
                print('nop')
        for (module_name, tg) in self.sparse_goals.items():
            for (target_name, target_goals) in tg.items():
                for (scheduled_key, goal) in target_goals.items():
                    setattr(self._target_spaces[module_name][target_name], scheduled_key, goal * ratio)

class _ComboPruner(ScheduledPruner):

    def __init__(self, pruner: Pruner, interval_steps: int, total_times: int, evaluator: Evaluator | None=None):
        if False:
            print('Hello World!')
        assert isinstance(pruner, Pruner)
        assert hasattr(pruner, 'interval_steps') and hasattr(pruner, 'total_times')
        if not isinstance(pruner, (LevelPruner, L1NormPruner, L2NormPruner, FPGMPruner, SlimPruner, TaylorPruner)):
            warning_msg = f'Compatibility not tested with pruner type {pruner.__class__.__name__}.'
            _logger.warning(warning_msg)
        if pruner._is_wrapped:
            pruner.unwrap_model()
        model = pruner.bound_model
        existed_wrappers = pruner._module_wrappers
        if pruner.evaluator is not None and evaluator is not None:
            _logger.warning('Pruner already has evaluator, the new evaluator passed to this function will be ignored.')
        evaluator = pruner.evaluator if pruner.evaluator else evaluator
        assert isinstance(evaluator, Evaluator)
        super().__init__(model=model, config_list=[], evaluator=evaluator, existed_wrappers=existed_wrappers)
        self.fused_compressors.extend(pruner.fused_compressors[1:])
        self._target_spaces = pruner._target_spaces
        self.interval_steps = interval_steps
        self.total_times = total_times
        self.bound_pruner = pruner
        self._init_sparse_goals()
        self._initial_ratio = 0.0

    @classmethod
    def from_compressor(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'{cls.__name__} can not initialized from any compressor.')

    def _initialize_state(self):
        if False:
            return 10
        self._update_sparse_goals_by_ratio(self._initial_ratio)
        self.bound_pruner.interval_steps = self.interval_steps
        self.bound_pruner.total_times = self.total_times

    def _register_trigger(self, evaluator: Evaluator):
        if False:
            return 10
        self._current_step = 0
        self._remaining_times = self.total_times

        def optimizer_task():
            if False:
                while True:
                    i = 10
            self._current_step += 1
            if self._current_step == self.interval_steps:
                self._remaining_times -= 1
                self.update_sparse_goals(self.total_times - self._remaining_times)
                debug_msg = f'{self.__class__.__name__} generate masks, remaining times {self._remaining_times}'
                _logger.debug(debug_msg)
                if self._remaining_times > 0:
                    self._current_step = 0
        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            while True:
                i = 10
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        if False:
            i = 10
            return i + 15
        self._initialize_state()
        self._register_trigger(evaluator)
        self.bound_pruner._fuse_preprocess(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        if False:
            return 10
        self.bound_pruner._fuse_postprocess(evaluator)

    def compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            return 10
        if max_steps is not None:
            assert max_steps >= self.total_times * self.interval_steps
        else:
            warn_msg = f'Using epochs number as training duration, ' + 'please make sure the total training steps larger than total_times * interval_steps.'
            _logger.warning(warn_msg)
        return super().compress(max_steps, max_epochs)

class LinearPruner(_ComboPruner):
    __doc__ = '\n    The sparse ratio or sparse threshold in the bound pruner will increase in a linear way from 0. to final::\n\n        current_sparse = (1 - initial_ratio) * current_times / total_times * final_sparse\n\n    If min/max sparse ratio is also set in target setting, they will also synchronous increase in a linear way.\n\n    Note that this pruner can not be initialized by ``LinearPruner.from_compressor(...)``.\n\n    Parameters\n    ----------\n    pruner\n        The bound pruner.\n    interval_steps\n        A integer number, for each ``interval_steps`` training, the sparse goal will be updated.\n    total_times\n        A integer number, how many times to update the sparse goal in total.\n    evaluator\n        {evaluator_docstring}\n\n    Examples\n    --------\n        Please refer to\n        :githublink:`examples/compression/pruning/scheduled_pruning.py <examples/compression/pruning/scheduled_pruning.py>`.\n    '.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    def update_sparse_goals(self, current_times: int):
        if False:
            while True:
                i = 10
        ratio = (1 - self._initial_ratio) * current_times / self.total_times
        self._update_sparse_goals_by_ratio(ratio)

class AGPPruner(_ComboPruner):
    __doc__ = '\n    The sparse ratio or sparse threshold in the bound pruner will increase in a AGP way from 0. to final::\n\n        current_sparse =  (1 - (1 - self._initial_ratio) * (1 - current_times / self.total_times) ** 3) * final_sparse\n\n    If min/max sparse ratio is also set in target setting, they will also synchronous increase in a AGP way.\n\n    Note that this pruner can not be initialized by ``AGPPruner.from_compressor(...)``.\n\n    Parameters\n    ----------\n    pruner\n        The bound pruner.\n    interval_steps\n        A integer number, for each ``interval_steps`` training, the sparse goal will be updated.\n    total_times\n        A integer number, how many times to update the sparse goal in total.\n    evaluator\n        {evaluator_docstring}\n\n    Examples\n    --------\n        Please refer to\n        :githublink:`examples/compression/pruning/scheduled_pruning.py <examples/compression/pruning/scheduled_pruning.py>`.\n    '.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    def update_sparse_goals(self, current_times: int):
        if False:
            while True:
                i = 10
        ratio = 1 - (1 - self._initial_ratio) * (1 - current_times / self.total_times) ** 3
        self._update_sparse_goals_by_ratio(ratio)