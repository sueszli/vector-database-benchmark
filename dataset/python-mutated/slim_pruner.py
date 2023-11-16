from __future__ import annotations
from collections import defaultdict
import logging
from typing import Dict, List, Literal, Tuple, overload
import torch
from torch.optim import Adam
from .tools import _METRICS, _MASKS, generate_sparsity, is_active_target
from ..base.compressor import Compressor, Pruner
from ..base.target_space import TargetType
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING
_logger = logging.getLogger(__name__)
SLIM_SCALING_FACTOR_PNAME = '{}_slim_factor'

class SlimPruner(Pruner):
    __doc__ = "\n    Slim pruner adds sparsity regularization on the scaling factors of batch normalization (BN) layers during training\n    to identify unimportant channels. The channels with small scaling factor values will be pruned.\n\n    For more details, please refer to\n    `Learning Efficient Convolutional Networks through Network Slimming <https://arxiv.org/abs/1708.06519>`__.\n\n    Parameters\n    ----------\n    model\n        Model to be pruned.\n    config_list\n        A list of dict, each dict configure which module need to be pruned, and how to prune.\n        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.\n    evaluator\n        {evaluator_docstring}\n    training_steps\n        An integer to control steps of training the model and scale factors. Masks will be generated after ``training_steps``.\n    regular_scale\n        ``regular_scale`` controls the scale factors' penalty.\n    \n    Examples\n    --------\n        Please refer to\n        :githublink:`examples/compression/pruning/slim_pruning.py <examples/compression/pruning/slim_pruning.py>`.\n    ".format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int, regular_scale: float=1.0):
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int, regular_scale: float=1.0, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            i = 10
            return i + 15
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int, regular_scale: float=1.0, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            return 10
        super().__init__(model, config_list, evaluator, existed_wrappers)
        self.evaluator: Evaluator
        self.training_steps = training_steps
        self.regular_scale = regular_scale
        self._current_step = 0
        self.interval_steps = training_steps
        self.total_times: int | Literal['unlimited'] = 1
        self._set_apply_method()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], training_steps: int, regular_scale: float=1.0, evaluator: Evaluator | None=None):
        if False:
            while True:
                i = 10
        return super().from_compressor(compressor, new_config_list, training_steps=training_steps, regular_scale=regular_scale, evaluator=evaluator)

    def _set_apply_method(self):
        if False:
            for i in range(10):
                print('nop')
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                if target_space.apply_method == 'mul':
                    target_space.apply_method = 'slim_mul'
                else:
                    assert target_space.apply_method == 'slim_mul'

    def _register_scaling_facotrs(self):
        if False:
            print('Hello World!')
        self.scaling_factors = defaultdict(dict)
        for (module_name, ts) in self._target_spaces.items():
            for (target_name, target_space) in ts.items():
                if is_active_target(target_space):
                    if target_space.type is TargetType.PARAMETER:
                        scaling_factor = torch.ones_like(target_space.target)
                        if target_space._scaler is not None:
                            scaling_factor = target_space._scaler.shrink(scaling_factor, keepdim=True)
                        target_space._wrapper.register_parameter(SLIM_SCALING_FACTOR_PNAME.format(target_name), torch.nn.Parameter(scaling_factor))
                        scaling_factor = target_space._get_wrapper_attr(SLIM_SCALING_FACTOR_PNAME.format(target_name))
                        self.scaling_factors[module_name][target_name] = scaling_factor
                    else:
                        raise NotImplementedError()

    def _register_factors_optimization(self, evaluator: Evaluator):
        if False:
            for i in range(10):
                print('nop')
        scaling_factors = []
        for (_, target_scaling_factor) in self.scaling_factors.items():
            for (_, scaling_factor) in target_scaling_factor.items():
                scaling_factors.append(scaling_factor)
        if not scaling_factors:
            return
        params = [{'params': scaling_factors}]
        optimizer = Adam(params, 0.01)
        evaluator.patch_optimizer_step(before_step_tasks=[optimizer.step], after_step_tasks=[optimizer.zero_grad])

    def _patch_loss(self, evaluator: Evaluator):
        if False:
            for i in range(10):
                print('nop')

        def loss_patch(original_loss, batch):
            if False:
                print('Hello World!')
            reg_loss = torch.tensor(0.0, device=original_loss.device)
            count = 0
            for (_, target_scaling_factor) in self.scaling_factors.items():
                for (_, scaling_factor) in target_scaling_factor.items():
                    reg_loss = reg_loss + scaling_factor.norm(p=1)
                    count += 1
            if count > 0:
                reg_loss = self.regular_scale * reg_loss / count
            return original_loss + reg_loss
        evaluator.patch_loss(loss_patch)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if False:
            while True:
                i = 10
        data = defaultdict(dict)
        for (module_name, ts) in self._target_spaces.items():
            for (target_name, target_space) in ts.items():
                scaling_factor: torch.Tensor = getattr(target_space._wrapper, SLIM_SCALING_FACTOR_PNAME.format(target_name), None)
                if scaling_factor is not None:
                    data[module_name][target_name] = scaling_factor.clone().detach()
        return data

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
        if False:
            print('Hello World!')
        return {k: {p: q.abs() for (p, q) in v.items()} for (k, v) in data.items()}

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        if False:
            for i in range(10):
                print('nop')
        return generate_sparsity(metrics, self._target_spaces)

    def _register_trigger(self, evaluator: Evaluator):
        if False:
            print('Hello World!')
        assert self.interval_steps >= self.training_steps or self.interval_steps < 0
        self._remaining_times = self.total_times

        def optimizer_task():
            if False:
                return 10
            self._current_step += 1
            if self._current_step == self.training_steps:
                masks = self.generate_masks()
                self.update_masks(masks)
                if isinstance(self._remaining_times, int):
                    self._remaining_times -= 1
                debug_msg = f'{self.__class__.__name__} generate masks, remaining times {self._remaining_times}'
                _logger.debug(debug_msg)
            if self._current_step == self.interval_steps and (self._remaining_times == 'unlimited' or self._remaining_times > 0):
                self._current_step = 0
        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            return 10
        assert max_steps is None and max_epochs is None
        self._fusion_compress(self.training_steps, None)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._register_scaling_facotrs()
        self._register_factors_optimization(evaluator)
        self._patch_loss(evaluator)
        self._register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        if False:
            while True:
                i = 10
        pass

    @overload
    def compress(self) -> Tuple[torch.nn.Module, _MASKS]:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def compress(self, max_steps: int | None, max_epochs: int | None) -> Tuple[torch.nn.Module, _MASKS]:
        if False:
            return 10
        ...

    def compress(self, max_steps: int | None=None, max_epochs: int | None=None):
        if False:
            while True:
                i = 10
        return super().compress(max_steps, max_epochs)