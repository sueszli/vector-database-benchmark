from collections import OrderedDict
from typing import Dict, Any
from torch.ao.quantization.utils import Pattern
from ..fake_quantize import FixedQParamsFakeQuantize
from ..observer import ObserverBase
import copy
__all__ = ['get_default_fusion_patterns', 'get_default_quant_patterns', 'get_default_output_activation_post_process_map']
QuantizeHandler = Any
_DEFAULT_FUSION_PATTERNS = OrderedDict()

def _register_fusion_pattern(pattern):
    if False:
        for i in range(10):
            print('nop')

    def insert(fn):
        if False:
            for i in range(10):
                print('nop')
        _DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn
    return insert

def get_default_fusion_patterns() -> Dict[Pattern, QuantizeHandler]:
    if False:
        print('Hello World!')
    return copy.copy(_DEFAULT_FUSION_PATTERNS)
_DEFAULT_QUANTIZATION_PATTERNS = OrderedDict()
_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP = {}
_DEFAULT_OUTPUT_OBSERVER_MAP = {}

def _register_quant_pattern(pattern, fixed_qparams_observer=None):
    if False:
        for i in range(10):
            print('nop')

    def insert(fn):
        if False:
            i = 10
            return i + 15
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if fixed_qparams_observer is not None:
            _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP[pattern] = FixedQParamsFakeQuantize.with_args(observer=fixed_qparams_observer)
            _DEFAULT_OUTPUT_OBSERVER_MAP[pattern] = fixed_qparams_observer
        return fn
    return insert

def get_default_quant_patterns() -> Dict[Pattern, QuantizeHandler]:
    if False:
        print('Hello World!')
    return copy.copy(_DEFAULT_QUANTIZATION_PATTERNS)

def get_default_output_activation_post_process_map(is_training) -> Dict[Pattern, ObserverBase]:
    if False:
        return 10
    if is_training:
        return copy.copy(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP)
    else:
        return copy.copy(_DEFAULT_OUTPUT_OBSERVER_MAP)

def _sorted_patterns_dict(patterns_dict: Dict[Pattern, QuantizeHandler]) -> Dict[Pattern, QuantizeHandler]:
    if False:
        print('Hello World!')
    '\n    Return a sorted version of the patterns dictionary such that longer patterns are matched first,\n    e.g. match (F.relu, F.linear) before F.relu.\n    This works for current use cases, but we may need to have a more clever way to sort\n    things to address more complex patterns\n    '

    def get_len(pattern):
        if False:
            while True:
                i = 10
        ' this will calculate the length of the pattern by counting all the entries\n        in the pattern.\n        this will make sure (nn.ReLU, (nn.BatchNorm, nn.Conv2d)) comes before\n        (nn.BatchNorm, nn.Conv2d) so that we can match the former first\n        '
        len = 0
        if isinstance(pattern, tuple):
            for item in pattern:
                len += get_len(item)
        else:
            len += 1
        return len
    return OrderedDict(sorted(patterns_dict.items(), key=lambda kv: -get_len(kv[0]) if isinstance(kv[0], tuple) else 1))