import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from ray.serve._private.common import ApplicationName, EndpointInfo, EndpointTag, RequestProtocol
from ray.serve._private.constants import RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING, SERVE_LOGGER_NAME
from ray.serve.handle import RayServeHandle
logger = logging.getLogger(SERVE_LOGGER_NAME)

class ProxyRouter(ABC):
    """Router interface for the proxy to use."""

    @abstractmethod
    def update_routes(self, endpoints: Dict[EndpointTag, EndpointInfo]):
        if False:
            return 10
        raise NotImplementedError

class LongestPrefixRouter(ProxyRouter):
    """Router that performs longest prefix matches on incoming routes."""

    def __init__(self, get_handle: Callable, protocol: RequestProtocol):
        if False:
            for i in range(10):
                print('nop')
        self._get_handle = get_handle
        self._protocol = protocol
        self.sorted_routes: List[str] = list()
        self.route_info: Dict[str, EndpointTag] = dict()
        self.handles: Dict[EndpointTag, RayServeHandle] = dict()
        self.app_to_is_cross_language: Dict[ApplicationName, bool] = dict()

    def update_routes(self, endpoints: Dict[EndpointTag, EndpointInfo]) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.info(f'Got updated endpoints: {endpoints}.', extra={'log_to_stderr': False})
        existing_handles = set(self.handles.keys())
        routes = []
        route_info = {}
        app_to_is_cross_language = {}
        for (endpoint, info) in endpoints.items():
            routes.append(info.route)
            route_info[info.route] = endpoint
            app_to_is_cross_language[endpoint.app] = info.app_is_cross_language
            if endpoint in self.handles:
                existing_handles.remove(endpoint)
            else:
                handle = self._get_handle(endpoint.name, endpoint.app).options(stream=not info.app_is_cross_language, use_new_handle_api=True, _prefer_local_routing=RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING)
                handle._set_request_protocol(self._protocol)
                self.handles[endpoint] = handle
        if len(existing_handles) > 0:
            logger.info(f'Deleting {len(existing_handles)} unused handles.', extra={'log_to_stderr': False})
        for endpoint in existing_handles:
            del self.handles[endpoint]
        self.sorted_routes = sorted(routes, key=lambda x: len(x), reverse=True)
        self.route_info = route_info
        self.app_to_is_cross_language = app_to_is_cross_language

    def match_route(self, target_route: str) -> Optional[Tuple[str, RayServeHandle, bool]]:
        if False:
            print('Hello World!')
        'Return the longest prefix match among existing routes for the route.\n        Args:\n            target_route: route to match against.\n        Returns:\n            (route, handle, is_cross_language) if found, else None.\n        '
        for route in self.sorted_routes:
            if target_route.startswith(route):
                matched = False
                if route.endswith('/'):
                    matched = True
                elif len(target_route) == len(route) or target_route[len(route)] == '/':
                    matched = True
                if matched:
                    endpoint = self.route_info[route]
                    return (route, self.handles[endpoint], self.app_to_is_cross_language[endpoint.app])
        return None

class EndpointRouter(ProxyRouter):
    """Router that matches endpoint to return the handle."""

    def __init__(self, get_handle: Callable, protocol: RequestProtocol):
        if False:
            return 10
        self._get_handle = get_handle
        self._protocol = protocol
        self.handles: Dict[EndpointTag, RayServeHandle] = dict()
        self.endpoints: Dict[EndpointTag, EndpointInfo] = dict()

    def update_routes(self, endpoints: Dict[EndpointTag, EndpointInfo]):
        if False:
            while True:
                i = 10
        logger.info(f'Got updated endpoints: {endpoints}.', extra={'log_to_stderr': False})
        self.endpoints = endpoints
        existing_handles = set(self.handles.keys())
        for (endpoint, info) in endpoints.items():
            if endpoint in self.handles:
                existing_handles.remove(endpoint)
            else:
                handle = self._get_handle(endpoint.name, endpoint.app).options(stream=not info.app_is_cross_language, use_new_handle_api=True, _prefer_local_routing=RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING)
                handle._set_request_protocol(self._protocol)
                self.handles[endpoint] = handle
        if len(existing_handles) > 0:
            logger.info(f'Deleting {len(existing_handles)} unused handles.', extra={'log_to_stderr': False})
        for endpoint in existing_handles:
            del self.handles[endpoint]

    def get_handle_for_endpoint(self, target_app_name: str) -> Optional[Tuple[str, RayServeHandle, bool]]:
        if False:
            while True:
                i = 10
        'Return the handle that matches with endpoint.\n\n        Args:\n            target_app_name: app_name to match against.\n        Returns:\n            (route, handle, app_name, is_cross_language) for the single app if there\n            is only one, else find the app and handle for exact match. Else return None.\n        '
        for (endpoint_tag, handle) in self.handles.items():
            if target_app_name == endpoint_tag.app or len(self.handles) == 1:
                endpoint_info = self.endpoints[endpoint_tag]
                return (endpoint_info.route, handle, endpoint_info.app_is_cross_language)
        return None