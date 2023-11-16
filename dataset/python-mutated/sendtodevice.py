import logging
from typing import TYPE_CHECKING, Tuple
from synapse.http import servlet
from synapse.http.server import HttpServer
from synapse.http.servlet import assert_params_in_dict, parse_json_object_from_request
from synapse.http.site import SynapseRequest
from synapse.logging.opentracing import set_tag
from synapse.rest.client.transactions import HttpTransactionCache
from synapse.types import JsonDict, Requester
from ._base import client_patterns
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class SendToDeviceRestServlet(servlet.RestServlet):
    PATTERNS = client_patterns('/sendToDevice/(?P<message_type>[^/]*)/(?P<txn_id>[^/]*)$')
    CATEGORY = 'The to_device stream'

    def __init__(self, hs: 'HomeServer'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.hs = hs
        self.auth = hs.get_auth()
        self.txns = HttpTransactionCache(hs)
        self.device_message_handler = hs.get_device_message_handler()

    async def on_PUT(self, request: SynapseRequest, message_type: str, txn_id: str) -> Tuple[int, JsonDict]:
        requester = await self.auth.get_user_by_req(request, allow_guest=True)
        set_tag('txn_id', txn_id)
        return await self.txns.fetch_or_execute_request(request, requester, self._put, request, requester, message_type)

    async def _put(self, request: SynapseRequest, requester: Requester, message_type: str) -> Tuple[int, JsonDict]:
        content = parse_json_object_from_request(request)
        assert_params_in_dict(content, ('messages',))
        await self.device_message_handler.send_device_message(requester, message_type, content['messages'])
        return (200, {})

def register_servlets(hs: 'HomeServer', http_server: HttpServer) -> None:
    if False:
        print('Hello World!')
    SendToDeviceRestServlet(hs).register(http_server)