import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Tuple
from synapse.api.errors import Codes, SynapseError
from synapse.http.server import HttpServer
from synapse.http.servlet import RestServlet, parse_strings_from_args
from synapse.http.site import SynapseRequest
from synapse.types import JsonDict
from ._base import client_patterns
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class UserMutualRoomsServlet(RestServlet):
    """
    GET /uk.half-shot.msc2666/user/mutual_rooms?user_id={user_id} HTTP/1.1
    """
    PATTERNS = client_patterns('/uk.half-shot.msc2666/user/mutual_rooms$', releases=())

    def __init__(self, hs: 'HomeServer'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.auth = hs.get_auth()
        self.store = hs.get_datastores().main

    async def on_GET(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        args: Dict[bytes, List[bytes]] = request.args
        user_ids = parse_strings_from_args(args, 'user_id', required=True)
        if len(user_ids) > 1:
            raise SynapseError(HTTPStatus.BAD_REQUEST, 'Duplicate user_id query parameter', errcode=Codes.INVALID_PARAM)
        if b'batch_token' in args:
            raise SynapseError(HTTPStatus.BAD_REQUEST, 'Unknown batch_token', errcode=Codes.INVALID_PARAM)
        user_id = user_ids[0]
        requester = await self.auth.get_user_by_req(request)
        if user_id == requester.user.to_string():
            raise SynapseError(HTTPStatus.UNPROCESSABLE_ENTITY, 'You cannot request a list of shared rooms with yourself', errcode=Codes.INVALID_PARAM)
        rooms = await self.store.get_mutual_rooms_between_users(frozenset((requester.user.to_string(), user_id)))
        return (200, {'joined': list(rooms)})

def register_servlets(hs: 'HomeServer', http_server: HttpServer) -> None:
    if False:
        i = 10
        return i + 15
    UserMutualRoomsServlet(hs).register(http_server)