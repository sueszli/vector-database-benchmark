import io
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar, Union
import pickle
import ray
from ray.dag.base import DAGNodeBase
_instances: Dict[int, '_PyObjScanner'] = {}
SourceType = TypeVar('SourceType')
TransformedType = TypeVar('TransformedType')

def _get_node(instance_id: int, node_index: int) -> SourceType:
    if False:
        while True:
            i = 10
    'Get the node instance.\n\n    Note: This function should be static and globally importable,\n    otherwise the serialization overhead would be very significant.\n    '
    return _instances[instance_id]._replace_index(node_index)

class _PyObjScanner(ray.cloudpickle.CloudPickler, Generic[SourceType, TransformedType]):
    """Utility to find and replace the `source_type` in Python objects.

    `source_type` can either be a single type or a tuple of multiple types.

    The caller must first call `find_nodes()`, then compute a replacement table and
    pass it to `replace_nodes`.

    This uses cloudpickle under the hood, so all sub-objects that are not `source_type`
    must be serializable.

    Args:
        source_type: the type(s) of object to find and replace. Default to DAGNodeBase.
    """

    def __init__(self, source_type: Union[Type, Tuple]=DAGNodeBase):
        if False:
            return 10
        self.source_type = source_type
        self._buf = io.BytesIO()
        self._found = None
        self._objects = []
        self._replace_table: Dict[SourceType, TransformedType] = None
        _instances[id(self)] = self
        super().__init__(self._buf)

    def reducer_override(self, obj):
        if False:
            return 10
        'Hook for reducing objects.\n\n        Objects of `self.source_type` are saved to `self._found` and a global map so\n        they can later be replaced.\n\n        All other objects fall back to the default `CloudPickler` serialization.\n        '
        if isinstance(obj, self.source_type):
            index = len(self._found)
            self._found.append(obj)
            return (_get_node, (id(self), index))
        return super().reducer_override(obj)

    def find_nodes(self, obj: Any) -> List[SourceType]:
        if False:
            while True:
                i = 10
        'Find top-level DAGNodes.'
        assert self._found is None, 'find_nodes cannot be called twice on the same PyObjScanner instance.'
        self._found = []
        self._objects = []
        self.dump(obj)
        return self._found

    def replace_nodes(self, table: Dict[SourceType, TransformedType]) -> Any:
        if False:
            i = 10
            return i + 15
        'Replace previously found DAGNodes per the given table.'
        assert self._found is not None, 'find_nodes must be called first'
        self._replace_table = table
        self._buf.seek(0)
        return pickle.load(self._buf)

    def _replace_index(self, i: int) -> SourceType:
        if False:
            i = 10
            return i + 15
        return self._replace_table[self._found[i]]

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Clear the scanner from the _instances'
        if id(self) in _instances:
            del _instances[id(self)]

    def __del__(self):
        if False:
            while True:
                i = 10
        self.clear()