import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Tuple
from synapse.api.room_versions import KNOWN_ROOM_VERSIONS, MSC3244_CAPABILITIES
from synapse.http.server import HttpServer
from synapse.http.servlet import RestServlet
from synapse.http.site import SynapseRequest
from synapse.types import JsonDict
from ._base import client_patterns
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class CapabilitiesRestServlet(RestServlet):
    """End point to expose the capabilities of the server."""
    PATTERNS = client_patterns('/capabilities$')
    CATEGORY = 'Client API requests'

    def __init__(self, hs: 'HomeServer'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.hs = hs
        self.config = hs.config
        self.auth = hs.get_auth()
        self.auth_handler = hs.get_auth_handler()

    async def on_GET(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        await self.auth.get_user_by_req(request, allow_guest=True)
        change_password = self.auth_handler.can_change_password()
        response: JsonDict = {'capabilities': {'m.room_versions': {'default': self.config.server.default_room_version.identifier, 'available': {v.identifier: v.disposition for v in KNOWN_ROOM_VERSIONS.values()}}, 'm.change_password': {'enabled': change_password}, 'm.set_displayname': {'enabled': self.config.registration.enable_set_displayname}, 'm.set_avatar_url': {'enabled': self.config.registration.enable_set_avatar_url}, 'm.3pid_changes': {'enabled': self.config.registration.enable_3pid_changes}, 'm.get_login_token': {'enabled': self.config.auth.login_via_existing_enabled}}}
        if self.config.experimental.msc3244_enabled:
            response['capabilities']['m.room_versions']['org.matrix.msc3244.room_capabilities'] = MSC3244_CAPABILITIES
        if self.config.experimental.msc3720_enabled:
            response['capabilities']['org.matrix.msc3720.account_status'] = {'enabled': True}
        if self.config.experimental.msc3664_enabled:
            response['capabilities']['im.nheko.msc3664.related_event_match'] = {'enabled': self.config.experimental.msc3664_enabled}
        return (HTTPStatus.OK, response)

def register_servlets(hs: 'HomeServer', http_server: HttpServer) -> None:
    if False:
        return 10
    CapabilitiesRestServlet(hs).register(http_server)