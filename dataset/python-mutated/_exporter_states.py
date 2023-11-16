from __future__ import annotations
from typing import Dict
from torch import _C

class ExportTypes:
    """Specifies how the ONNX model is stored."""
    PROTOBUF_FILE = 'Saves model in the specified protobuf file.'
    ZIP_ARCHIVE = 'Saves model in the specified ZIP file (uncompressed).'
    COMPRESSED_ZIP_ARCHIVE = 'Saves model in the specified ZIP file (compressed).'
    DIRECTORY = 'Saves model in the specified folder.'

class SymbolicContext:
    """Extra context for symbolic functions.

    Args:
        params_dict (Dict[str, _C.IValue]): Mapping from graph initializer name to IValue.
        env (Dict[_C.Value, _C.Value]): Mapping from Torch domain graph Value to ONNX domain graph Value.
        cur_node (_C.Node): Current node being converted to ONNX domain.
        onnx_block (_C.Block): Current ONNX block that converted nodes are being appended to.
    """

    def __init__(self, params_dict: Dict[str, _C.IValue], env: dict, cur_node: _C.Node, onnx_block: _C.Block):
        if False:
            print('Hello World!')
        self.params_dict: Dict[str, _C.IValue] = params_dict
        self.env: Dict[_C.Value, _C.Value] = env
        self.cur_node: _C.Node = cur_node
        self.onnx_block: _C.Block = onnx_block