import functools
import logging
from typing import Any, Callable, Dict, Optional, Union
from fastapi import Depends, FastAPI
from ray import cloudpickle, serve
from ray.serve._private.constants import DAG_DEPRECATION_MESSAGE, SERVE_LOGGER_NAME
from ray.serve._private.http_util import ASGIAppReplicaWrapper
from ray.serve._private.usage import ServeUsageTag
from ray.serve.deployment_graph import RayServeDAGHandle
from ray.serve.drivers_utils import load_http_adapter
from ray.serve.exceptions import RayServeException
from ray.serve.handle import RayServeHandle
from ray.util.annotations import Deprecated
logger = logging.getLogger(SERVE_LOGGER_NAME)

@Deprecated(message=DAG_DEPRECATION_MESSAGE)
@serve.deployment
class DAGDriver(ASGIAppReplicaWrapper):
    """A driver implementation that accepts HTTP requests."""
    MATCH_ALL_ROUTE_PREFIX = '/{path:path}'

    def __init__(self, dags: Union[RayServeDAGHandle, Dict[str, RayServeDAGHandle]], http_adapter: Optional[Union[str, Callable]]=None):
        if False:
            print('Hello World!')
        'Create a DAGDriver.\n\n        Args:\n            dags: a handle to a Ray Serve DAG or a dictionary of handles.\n            http_adapter: a callable function or import string to convert\n                HTTP requests to Ray Serve input.\n        '
        ServeUsageTag.DAG_DRIVER_USED.record('1')
        if http_adapter is not None:
            ServeUsageTag.HTTP_ADAPTER_USED.record('1')
        http_adapter = load_http_adapter(http_adapter)
        app = FastAPI()
        new_dags = {}
        if isinstance(dags, dict):
            for (route, handle) in dags.items():
                if isinstance(handle, RayServeHandle):
                    handle = handle.options(use_new_handle_api=True)
                new_dags[route] = handle

                def endpoint_create(route):
                    if False:
                        while True:
                            i = 10

                    @app.get(f'{route}')
                    @app.post(f'{route}')
                    async def handle_request(inp=Depends(http_adapter)):
                        return await self.predict_with_route(route, inp)
                endpoint_create_func = functools.partial(endpoint_create, route)
                endpoint_create_func()
        else:
            if isinstance(dags, RayServeHandle):
                dags = dags.options(use_new_handle_api=True)
            new_dags[self.MATCH_ALL_ROUTE_PREFIX] = dags

            @app.get(self.MATCH_ALL_ROUTE_PREFIX)
            @app.post(self.MATCH_ALL_ROUTE_PREFIX)
            async def handle_request(inp=Depends(http_adapter)):
                return await self.predict(inp)
        self.dags = new_dags
        frozen_app = cloudpickle.loads(cloudpickle.dumps(app))
        super().__init__(frozen_app)

    async def predict(self, *args, _ray_cache_refs: bool=False, **kwargs):
        """Perform inference directly without HTTP."""
        dag = self.dags[self.MATCH_ALL_ROUTE_PREFIX]
        if isinstance(dag, RayServeDAGHandle):
            kwargs['_ray_cache_refs'] = _ray_cache_refs
            return await dag.remote(*args, **kwargs)
        else:
            return await dag.remote(*args, **kwargs)

    async def predict_with_route(self, route_path, *args, **kwargs):
        """Perform inference directly without HTTP for multi dags."""
        if route_path not in self.dags:
            raise RayServeException(f'{route_path} does not exist in dags routes')
        dag = self.dags[route_path]
        return await dag.remote(*args, **kwargs)

    async def get_intermediate_object_refs(self) -> Dict[str, Any]:
        """Gets latest cached object refs from latest call to predict().

        Gets the latest cached references to the results of the default executors on
        each node in the DAG found at self.MATCH_ALL_ROUTE_PREFIX. Should be called
        after predict() has been called with _cache_refs set to True.
        """
        dag_handle = self.dags[self.MATCH_ALL_ROUTE_PREFIX]
        root_dag_node = dag_handle.dag_node
        if root_dag_node is None:
            raise AssertionError('Predict has not been called. Cannot retrieve intermediate object refs.')
        return await root_dag_node.get_object_refs_from_last_execute()

    async def get_pickled_dag_node(self) -> bytes:
        """Returns the serialized root dag node."""
        return self.dags[self.MATCH_ALL_ROUTE_PREFIX].pickled_dag_node