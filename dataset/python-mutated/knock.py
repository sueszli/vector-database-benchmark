import logging
from typing import TYPE_CHECKING, Dict, List, Tuple
from synapse.api.constants import Membership
from synapse.api.errors import SynapseError
from synapse.http.server import HttpServer
from synapse.http.servlet import RestServlet, parse_json_object_from_request, parse_strings_from_args
from synapse.http.site import SynapseRequest
from synapse.types import JsonDict, RoomAlias, RoomID
if TYPE_CHECKING:
    from synapse.app.homeserver import HomeServer
from ._base import client_patterns
logger = logging.getLogger(__name__)

class KnockRoomAliasServlet(RestServlet):
    """
    POST /knock/{roomIdOrAlias}
    """
    PATTERNS = client_patterns('/knock/(?P<room_identifier>[^/]*)')
    CATEGORY = 'Event sending requests'

    def __init__(self, hs: 'HomeServer'):
        if False:
            return 10
        super().__init__()
        self.room_member_handler = hs.get_room_member_handler()
        self.auth = hs.get_auth()

    async def on_POST(self, request: SynapseRequest, room_identifier: str) -> Tuple[int, JsonDict]:
        requester = await self.auth.get_user_by_req(request)
        content = parse_json_object_from_request(request)
        event_content = None
        if 'reason' in content:
            event_content = {'reason': content['reason']}
        if RoomID.is_valid(room_identifier):
            room_id = room_identifier
            args: Dict[bytes, List[bytes]] = request.args
            remote_room_hosts = parse_strings_from_args(args, 'server_name', required=False)
        elif RoomAlias.is_valid(room_identifier):
            handler = self.room_member_handler
            room_alias = RoomAlias.from_string(room_identifier)
            (room_id_obj, remote_room_hosts) = await handler.lookup_room_alias(room_alias)
            room_id = room_id_obj.to_string()
        else:
            raise SynapseError(400, '%s was not legal room ID or room alias' % (room_identifier,))
        await self.room_member_handler.update_membership(requester=requester, target=requester.user, room_id=room_id, action=Membership.KNOCK, third_party_signed=None, remote_room_hosts=remote_room_hosts, content=event_content)
        return (200, {'room_id': room_id})

def register_servlets(hs: 'HomeServer', http_server: HttpServer) -> None:
    if False:
        return 10
    KnockRoomAliasServlet(hs).register(http_server)