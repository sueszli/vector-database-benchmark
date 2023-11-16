import ray
from ray import cloudpickle
from ray.dag import DAGNode
from ray.dag.class_node import ClassNode
from ray.dag.function_node import FunctionNode
from ray.dag.input_node import InputNode
from ray.serve._private.constants import DAG_DEPRECATION_MESSAGE
from ray.util.annotations import Deprecated

@Deprecated(DAG_DEPRECATION_MESSAGE)
class RayServeDAGHandle:
    """Resolved from a DeploymentNode at runtime.

    This can be used to call the DAG from a driver deployment to efficiently
    orchestrate a deployment graph.
    """

    def __init__(self, pickled_dag_node: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.pickled_dag_node = pickled_dag_node
        self.dag_node = None

    @classmethod
    def _deserialize(cls, *args):
        if False:
            print('Hello World!')
        "Required for this class's __reduce__ method to be pickleable."
        return cls(*args)

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        return (RayServeDAGHandle._deserialize, (self.pickled_dag_node,))

    async def remote(self, *args, _ray_cache_refs: bool=False, **kwargs) -> ray.ObjectRef:
        """Execute the request, returns a ObjectRef representing final result."""
        if self.dag_node is None:
            self.dag_node = cloudpickle.loads(self.pickled_dag_node)
        return await self.dag_node.execute(*args, _ray_cache_refs=_ray_cache_refs, **kwargs)