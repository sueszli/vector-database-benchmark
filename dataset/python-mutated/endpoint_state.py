import logging
from typing import Any, Dict, Optional
from ray import cloudpickle
from ray.serve._private.common import EndpointInfo, EndpointTag
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
CHECKPOINT_KEY = 'serve-endpoint-state-checkpoint'
logger = logging.getLogger(SERVE_LOGGER_NAME)

class EndpointState:
    """Manages all state for endpoints in the system.

    This class is *not* thread safe, so any state-modifying methods should be
    called with a lock held.
    """

    def __init__(self, kv_store: KVStoreBase, long_poll_host: LongPollHost):
        if False:
            while True:
                i = 10
        self._kv_store = kv_store
        self._long_poll_host = long_poll_host
        self._endpoints: Dict[EndpointTag, EndpointInfo] = dict()
        checkpoint = self._kv_store.get(CHECKPOINT_KEY)
        if checkpoint is not None:
            self._endpoints = cloudpickle.loads(checkpoint)
        self._notify_route_table_changed()

    def shutdown(self):
        if False:
            return 10
        self._kv_store.delete(CHECKPOINT_KEY)

    def is_ready_for_shutdown(self) -> bool:
        if False:
            while True:
                i = 10
        'Returns whether the endpoint checkpoint has been deleted.\n\n        Get the endpoint checkpoint from the kv store. If it is None, then it has been\n        deleted.\n        '
        return self._kv_store.get(CHECKPOINT_KEY) is None

    def _checkpoint(self):
        if False:
            print('Hello World!')
        self._kv_store.put(CHECKPOINT_KEY, cloudpickle.dumps(self._endpoints))

    def _notify_route_table_changed(self):
        if False:
            while True:
                i = 10
        self._long_poll_host.notify_changed(LongPollNamespace.ROUTE_TABLE, self._endpoints)

    def _get_endpoint_for_route(self, route: str) -> Optional[EndpointTag]:
        if False:
            for i in range(10):
                print('nop')
        for (endpoint, info) in self._endpoints.items():
            if info.route == route:
                return endpoint
        return None

    def update_endpoint(self, endpoint: EndpointTag, endpoint_info: EndpointInfo) -> None:
        if False:
            while True:
                i = 10
        'Create or update the given endpoint.\n\n        This method is idempotent - if the endpoint already exists it will be\n        updated to match the given parameters. Calling this twice with the same\n        arguments is a no-op.\n        '
        if self._endpoints.get(endpoint) == endpoint_info:
            return
        existing_route_endpoint = self._get_endpoint_for_route(endpoint_info.route)
        if existing_route_endpoint is not None and existing_route_endpoint != endpoint:
            logger.debug(f'route_prefix "{endpoint_info.route}" is currently registered to deployment "{existing_route_endpoint.name}". Re-registering route_prefix "{endpoint_info.route}" to deployment "{endpoint.name}".')
            del self._endpoints[existing_route_endpoint]
        self._endpoints[endpoint] = endpoint_info
        self._checkpoint()
        self._notify_route_table_changed()

    def get_endpoint_route(self, endpoint: EndpointTag) -> Optional[str]:
        if False:
            while True:
                i = 10
        if endpoint in self._endpoints:
            return self._endpoints[endpoint].route
        return None

    def get_endpoints(self) -> Dict[EndpointTag, Dict[str, Any]]:
        if False:
            print('Hello World!')
        endpoints = {}
        for (endpoint, info) in self._endpoints.items():
            endpoints[endpoint] = {'route': info.route}
        return endpoints

    def delete_endpoint(self, endpoint: EndpointTag) -> None:
        if False:
            return 10
        if endpoint not in self._endpoints:
            return
        del self._endpoints[endpoint]
        self._checkpoint()
        self._notify_route_table_changed()