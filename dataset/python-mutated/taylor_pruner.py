from __future__ import annotations
from collections import defaultdict
import functools
import logging
from typing import Callable, Dict, List, Literal, Tuple, overload
import torch
from .tools import _METRICS, _MASKS, norm_metrics, generate_sparsity, is_active_target
from ..base.compressor import Compressor, Pruner
from ..base.target_space import TargetType
from ..base.wrapper import ModuleWrapper
from ..utils.docstring import _EVALUATOR_DOCSTRING
from ..utils.evaluator import Evaluator, TensorHook
_logger = logging.getLogger(__name__)

class TaylorPruner(Pruner):
    __doc__ = '\n    Taylor pruner is a pruner which prunes on the first weight dimension,\n    based on estimated importance calculated from the first order taylor expansion on weights to achieve a preset level of network sparsity.\n    The estimated importance is defined as the paper `Importance Estimation for Neural Network Pruning <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__.\n\n    :math:`\\widehat{\\mathcal{I}}_{\\mathcal{S}}^{(1)}(\\mathbf{W}) \\triangleq \\sum_{s \\in \\mathcal{S}} \\mathcal{I}_{s}^{(1)}(\\mathbf{W})=\\sum_{s \\in \\mathcal{S}}\\left(g_{s} w_{s}\\right)^{2}`\n    ' + '\n\n    Parameters\n    ----------\n    model\n        Model to be pruned.\n    config_list\n        A list of dict, each dict configure which module need to be pruned, and how to prune.\n        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.\n    evaluator\n        {evaluator_docstring}\n    training_steps\n        The step number used to collect gradients, the masks will be generated after training_steps training.\n\n    Examples\n    --------\n        Please refer to\n        :githublink:`examples/compression/pruning/taylor_pruning.py <examples/compression/pruning/taylor_pruning.py>`.\n    '.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int):
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int, existed_wrappers: Dict[str, ModuleWrapper]):
        if False:
            print('Hello World!')
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            return 10
        super().__init__(model=model, config_list=config_list, evaluator=evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.training_steps = training_steps
        self._current_step = 0
        self.hooks: Dict[str, Dict[str, TensorHook]] = defaultdict(dict)
        self.interval_steps = training_steps
        self.total_times: int | Literal['unlimited'] = 1

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], training_steps: int, evaluator: Evaluator | None=None):
        if False:
            print('Hello World!')
        return super().from_compressor(compressor, new_config_list, training_steps=training_steps, evaluator=evaluator)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if False:
            return 10
        data = defaultdict(dict)
        for (module_name, hooks) in self.hooks.items():
            for (target_name, hook) in hooks.items():
                if len(hook.buffer) > 0:
                    data[module_name][target_name] = hook.buffer[0] / self.training_steps
        return data

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
        if False:
            while True:
                i = 10
        return norm_metrics(p=1, data=data, target_spaces=self._target_spaces)

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        if False:
            return 10
        return generate_sparsity(metrics, self._target_spaces)

    def _register_hooks(self, evaluator: Evaluator):
        if False:
            for i in range(10):
                print('nop')

        def collector(buffer: List, target: torch.Tensor) -> Callable[[torch.Tensor], None]:
            if False:
                print('Hello World!')
            assert len(buffer) == 0, 'Buffer pass to taylor pruner collector is not empty.'

            def collect_taylor(grad: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                if len(buffer) == 0:
                    buffer.append(torch.zeros_like(grad))
                if self._current_step < self.training_steps:
                    buffer[0] += (target.detach() * grad.detach()).pow(2)
            return collect_taylor
        hook_list = []
        for (module_name, ts) in self._target_spaces.items():
            for (target_name, target_space) in ts.items():
                if is_active_target(target_space):
                    if target_space.type is TargetType.PARAMETER:
                        assert target_space.target is not None
                        hook = TensorHook(target_space.target, target_name, functools.partial(collector, target=target_space.target))
                        hook_list.append(hook)
                        self.hooks[module_name][target_name] = hook
                    else:
                        raise NotImplementedError()
        evaluator.register_hooks(hook_list)

    def _register_trigger(self, evaluator: Evaluator):
        if False:
            for i in range(10):
                print('nop')
        assert self.interval_steps >= self.training_steps or self.interval_steps < 0
        self._remaining_times = self.total_times

        def optimizer_task():
            if False:
                i = 10
                return i + 15
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
                for (_, hooks) in self.hooks.items():
                    for (_, hook) in hooks.items():
                        hook.buffer.clear()
        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            print('Hello World!')
        assert max_steps is None and max_epochs is None
        self._fusion_compress(self.training_steps, None)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._register_hooks(evaluator)
        self._register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @overload
    def compress(self) -> Tuple[torch.nn.Module, _MASKS]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def compress(self, max_steps: int | None, max_epochs: int | None) -> Tuple[torch.nn.Module, _MASKS]:
        if False:
            i = 10
            return i + 15
        ...

    def compress(self, max_steps: int | None=None, max_epochs: int | None=None):
        if False:
            while True:
                i = 10
        return super().compress(max_steps, max_epochs)