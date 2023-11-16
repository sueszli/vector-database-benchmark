import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Tuple
from synapse.http.servlet import RestServlet, parse_string
from synapse.http.site import SynapseRequest
from synapse.rest.admin._base import admin_patterns, assert_requester_is_admin
from synapse.types import JsonDict
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class UsernameAvailableRestServlet(RestServlet):
    """An admin API to check if a given username is available, regardless of whether registration is enabled.

    Example:
        GET /_synapse/admin/v1/username_available?username=foo
        200 OK
        {
            "available": true
        }
    """
    PATTERNS = admin_patterns('/username_available$')

    def __init__(self, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        self.auth = hs.get_auth()
        self.registration_handler = hs.get_registration_handler()

    async def on_GET(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        await assert_requester_is_admin(self.auth, request)
        username = parse_string(request, 'username', required=True)
        await self.registration_handler.check_username(username)
        return (HTTPStatus.OK, {'available': True})