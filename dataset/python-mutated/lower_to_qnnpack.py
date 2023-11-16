from ._lower_to_native_backend import _lower_to_native_backend
from ..qconfig import QConfigAny
from torch.fx import GraphModule
from typing import Dict, Tuple
__all__ = ['lower_to_qnnpack']

def lower_to_qnnpack(model: GraphModule, qconfig_map: Dict[str, QConfigAny], node_name_to_scope: Dict[str, Tuple[str, type]]) -> GraphModule:
    if False:
        return 10
    ' Lower a quantized reference model (with reference quantized operator patterns)\n    to qnnpack\n    '
    return _lower_to_native_backend(model, qconfig_map, node_name_to_scope)