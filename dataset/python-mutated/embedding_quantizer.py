from __future__ import annotations
import copy
from typing import List, Set
import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation, QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OperatorConfig, OperatorPatternType, QuantizationConfig
__all__ = ['get_embedding_operators_config', 'EmbeddingQuantizer']

def get_embedding_operators_config() -> OperatorConfig:
    if False:
        return 10
    weight_quantization_spec = QuantizationSpec(dtype=torch.uint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0, observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2 ** (-12)))
    quantization_config = QuantizationConfig(None, None, weight_quantization_spec, None)
    ops: List[OperatorPatternType] = [[torch.nn.Embedding]]
    ops.append([F.embedding])
    supported_config_and_operators = OperatorConfig(config=quantization_config, operators=ops)
    return copy.deepcopy(supported_config_and_operators)

class EmbeddingQuantizer(Quantizer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        if False:
            for i in range(10):
                print('nop')
        op_configs: Set[QuantizationConfig] = set({})
        for (spec, _) in cls.get_supported_operators():
            op_configs.add(spec)
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(cls, quantization_config: QuantizationConfig) -> List[OperatorPatternType]:
        if False:
            print('Hello World!')
        for (config, ops) in cls.get_supported_operators():
            if config == quantization_config:
                return ops
        return []

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        if False:
            i = 10
            return i + 15
        'just handling global spec for now'
        self._annotate_embedding_ops(model.graph)
        return model

    def _annotate_embedding_ops(self, graph: torch.fx.Graph) -> None:
        if False:
            for i in range(10):
                print('nop')
        embedding_config: OperatorConfig = get_embedding_operators_config()
        for node in graph.nodes:
            if node.op == 'call_function' and node.target == torch.ops.aten.embedding.default:
                if embedding_config.config.weight is None:
                    raise ValueError('Embedding config must have a valid weight quantization spec.')
                node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={node.args[0]: embedding_config.config.weight})

    def validate(self, model: torch.fx.GraphModule) -> None:
        if False:
            while True:
                i = 10
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        if False:
            while True:
                i = 10
        return [get_embedding_operators_config()]