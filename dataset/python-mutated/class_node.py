import ray
from ray.dag.dag_node import DAGNode
from ray.dag.input_node import InputNode
from ray.dag.format_utils import get_dag_node_str
from ray.dag.constants import PARENT_CLASS_NODE_KEY, PREV_CLASS_METHOD_CALL_KEY
from ray.util.annotations import DeveloperAPI
from typing import Any, Dict, List, Optional, Tuple

@DeveloperAPI
class ClassNode(DAGNode):
    """Represents an actor creation in a Ray task DAG."""

    def __init__(self, cls, cls_args, cls_kwargs, cls_options, other_args_to_resolve=None):
        if False:
            i = 10
            return i + 15
        self._body = cls
        self._last_call: Optional['ClassMethodNode'] = None
        super().__init__(cls_args, cls_kwargs, cls_options, other_args_to_resolve=other_args_to_resolve)
        if self._contains_input_node():
            raise ValueError('InputNode handles user dynamic input the the DAG, and cannot be used as args, kwargs, or other_args_to_resolve in ClassNode constructor because it is not available at class construction or binding time.')

    def _copy_impl(self, new_args: List[Any], new_kwargs: Dict[str, Any], new_options: Dict[str, Any], new_other_args_to_resolve: Dict[str, Any]):
        if False:
            print('Hello World!')
        return ClassNode(self._body, new_args, new_kwargs, new_options, other_args_to_resolve=new_other_args_to_resolve)

    def _execute_impl(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Executor of ClassNode by ray.remote()\n\n        Args and kwargs are to match base class signature, but not in the\n        implementation. All args and kwargs should be resolved and replaced\n        with value in bound_args and bound_kwargs via bottom-up recursion when\n        current node is executed.\n        '
        return ray.remote(self._body).options(**self._bound_options).remote(*self._bound_args, **self._bound_kwargs)

    def _contains_input_node(self) -> bool:
        if False:
            while True:
                i = 10
        'Check if InputNode is used in children DAGNodes with current node\n        as the root.\n        '
        children_dag_nodes = self._get_all_child_nodes()
        for child in children_dag_nodes:
            if isinstance(child, InputNode):
                return True
        return False

    def __getattr__(self, method_name: str):
        if False:
            print('Hello World!')
        if method_name == 'bind' and 'bind' not in dir(self._body):
            raise AttributeError(f'.bind() cannot be used again on {type(self)} ')
        getattr(self._body, method_name)
        call_node = _UnboundClassMethodNode(self, method_name)
        return call_node

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return get_dag_node_str(self, str(self._body))

class _UnboundClassMethodNode(object):

    def __init__(self, actor: ClassNode, method_name: str):
        if False:
            while True:
                i = 10
        self._actor = actor
        self._method_name = method_name
        self._options = {}

    def bind(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        other_args_to_resolve = {PARENT_CLASS_NODE_KEY: self._actor, PREV_CLASS_METHOD_CALL_KEY: self._actor._last_call}
        node = ClassMethodNode(self._method_name, args, kwargs, self._options, other_args_to_resolve=other_args_to_resolve)
        self._actor._last_call = node
        return node

    def __getattr__(self, attr: str):
        if False:
            print('Hello World!')
        if attr == 'remote':
            raise AttributeError('.remote() cannot be used on ClassMethodNodes. Use .bind() instead to express an symbolic actor call.')
        else:
            return self.__getattribute__(attr)

    def options(self, **options):
        if False:
            i = 10
            return i + 15
        self._options = options
        return self

@DeveloperAPI
class ClassMethodNode(DAGNode):
    """Represents an actor method invocation in a Ray function DAG."""

    def __init__(self, method_name: str, method_args: Tuple[Any], method_kwargs: Dict[str, Any], method_options: Dict[str, Any], other_args_to_resolve: Dict[str, Any]):
        if False:
            i = 10
            return i + 15
        self._bound_args = method_args or []
        self._bound_kwargs = method_kwargs or {}
        self._bound_options = method_options or {}
        self._method_name: str = method_name
        self._parent_class_node: ClassNode = other_args_to_resolve.get(PARENT_CLASS_NODE_KEY)
        self._prev_class_method_call: Optional[ClassMethodNode] = other_args_to_resolve.get(PREV_CLASS_METHOD_CALL_KEY, None)
        super().__init__(method_args, method_kwargs, method_options, other_args_to_resolve=other_args_to_resolve)

    def _copy_impl(self, new_args: List[Any], new_kwargs: Dict[str, Any], new_options: Dict[str, Any], new_other_args_to_resolve: Dict[str, Any]):
        if False:
            while True:
                i = 10
        return ClassMethodNode(self._method_name, new_args, new_kwargs, new_options, other_args_to_resolve=new_other_args_to_resolve)

    def _execute_impl(self, *args, **kwargs):
        if False:
            return 10
        'Executor of ClassMethodNode by ray.remote()\n\n        Args and kwargs are to match base class signature, but not in the\n        implementation. All args and kwargs should be resolved and replaced\n        with value in bound_args and bound_kwargs via bottom-up recursion when\n        current node is executed.\n        '
        method_body = getattr(self._parent_class_node, self._method_name)
        return method_body.options(**self._bound_options).remote(*self._bound_args, **self._bound_kwargs)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return get_dag_node_str(self, f'{self._method_name}()')

    def get_method_name(self) -> str:
        if False:
            print('Hello World!')
        return self._method_name