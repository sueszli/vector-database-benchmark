import logging
from typing import TYPE_CHECKING, Tuple
from synapse.handlers.device import DeviceHandler
from synapse.http.server import HttpServer
from synapse.http.servlet import RestServlet
from synapse.http.site import SynapseRequest
from synapse.rest.client._base import client_patterns
from synapse.types import JsonDict
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class LogoutRestServlet(RestServlet):
    PATTERNS = client_patterns('/logout$', v1=True)

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.auth = hs.get_auth()
        self._auth_handler = hs.get_auth_handler()
        handler = hs.get_device_handler()
        assert isinstance(handler, DeviceHandler)
        self._device_handler = handler

    async def on_POST(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        requester = await self.auth.get_user_by_req(request, allow_expired=True, allow_locked=True)
        if requester.device_id is None:
            access_token = self.auth.get_access_token_from_request(request)
            await self._auth_handler.delete_access_token(access_token)
        else:
            await self._device_handler.delete_devices(requester.user.to_string(), [requester.device_id])
        return (200, {})

class LogoutAllRestServlet(RestServlet):
    PATTERNS = client_patterns('/logout/all$', v1=True)

    def __init__(self, hs: 'HomeServer'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.auth = hs.get_auth()
        self._auth_handler = hs.get_auth_handler()
        handler = hs.get_device_handler()
        assert isinstance(handler, DeviceHandler)
        self._device_handler = handler

    async def on_POST(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        requester = await self.auth.get_user_by_req(request, allow_expired=True, allow_locked=True)
        user_id = requester.user.to_string()
        await self._device_handler.delete_all_devices_for_user(user_id)
        await self._auth_handler.delete_access_tokens_for_user(user_id)
        return (200, {})

def register_servlets(hs: 'HomeServer', http_server: HttpServer) -> None:
    if False:
        print('Hello World!')
    if hs.config.experimental.msc3861.enabled:
        return
    LogoutRestServlet(hs).register(http_server)
    LogoutAllRestServlet(hs).register(http_server)