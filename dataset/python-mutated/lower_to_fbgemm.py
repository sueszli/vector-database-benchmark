from ._lower_to_native_backend import _lower_to_native_backend
from ..qconfig import QConfigAny
from torch.fx import GraphModule
from typing import Dict, Tuple
__all__ = ['lower_to_fbgemm']

def lower_to_fbgemm(model: GraphModule, qconfig_map: Dict[str, QConfigAny], node_name_to_scope: Dict[str, Tuple[str, type]]) -> GraphModule:
    if False:
        i = 10
        return i + 15
    ' Lower a quantized reference model (with reference quantized operator patterns)\n    to fbgemm\n    '
    return _lower_to_native_backend(model, qconfig_map, node_name_to_scope)