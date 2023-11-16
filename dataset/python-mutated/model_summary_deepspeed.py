"""Utilities that can be used with Deepspeed."""
from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from lightning.pytorch.utilities.model_summary.model_summary import NOT_APPLICABLE, LayerSummary, ModelSummary, _is_lazy_weight_tensor, get_human_readable_count

def deepspeed_param_size(p: torch.nn.Parameter) -> int:
    if False:
        for i in range(10):
            print('nop')
    assert hasattr(p, 'ds_numel')
    return p.ds_numel

class DeepSpeedLayerSummary(LayerSummary):

    @property
    def num_parameters(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the number of parameters in this module.'
        return sum((deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._module.parameters()))

    @property
    def average_shard_parameters(self) -> int:
        if False:
            while True:
                i = 10
        'Returns the number of parameters in this module.'

        def partitioned_size(p: Parameter) -> int:
            if False:
                return 10
            return p.partitioned_size() if RequirementCache('deepspeed<0.6.6') else p.partition_numel()
        return sum((partitioned_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._module.parameters()))

class DeepSpeedSummary(ModelSummary):

    def summarize(self) -> Dict[str, DeepSpeedLayerSummary]:
        if False:
            i = 10
            return i + 15
        summary = OrderedDict(((name, DeepSpeedLayerSummary(module)) for (name, module) in self.named_modules))
        if self._model.example_input_array is not None:
            self._forward_example_input()
        for layer in summary.values():
            layer.detach_hook()
        if self._max_depth >= 1:
            for k in [k for k in summary if k.count('.') >= self._max_depth]:
                del summary[k]
        return summary

    @property
    def total_parameters(self) -> int:
        if False:
            while True:
                i = 10
        return sum((deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._model.parameters()))

    @property
    def trainable_parameters(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return sum((deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._model.parameters() if p.requires_grad))

    @property
    def parameters_per_layer(self) -> List[int]:
        if False:
            print('Hello World!')
        return [layer.average_shard_parameters for layer in self._layer_summary.values()]

    def _get_summary_data(self) -> List[Tuple[str, List[str]]]:
        if False:
            while True:
                i = 10
        'Makes a summary listing with:\n\n        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size\n\n        '
        arrays = [(' ', list(map(str, range(len(self._layer_summary))))), ('Name', self.layer_names), ('Type', self.layer_types), ('Params', list(map(get_human_readable_count, self.param_nums))), ('Params per Device', list(map(get_human_readable_count, self.parameters_per_layer)))]
        if self._model.example_input_array is not None:
            arrays.append(('In sizes', [str(x) for x in self.in_sizes]))
            arrays.append(('Out sizes', [str(x) for x in self.out_sizes]))
        total_leftover_params = self.total_parameters - self.total_layer_params
        if total_leftover_params > 0:
            self._add_leftover_params_to_summary(arrays, total_leftover_params)
        return arrays

    def _add_leftover_params_to_summary(self, arrays: List[Tuple[str, List[str]]], total_leftover_params: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add summary of params not associated with module or layer to model summary.'
        super()._add_leftover_params_to_summary(arrays, total_leftover_params)
        layer_summaries = dict(arrays)
        layer_summaries['Params per Device'].append(NOT_APPLICABLE)