import logging
from typing import TYPE_CHECKING
from synapse.http.server import DirectServeHtmlResource
from synapse.http.site import SynapseRequest
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class OIDCCallbackResource(DirectServeHtmlResource):
    isLeaf = 1

    def __init__(self, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        super().__init__()
        self._oidc_handler = hs.get_oidc_handler()

    async def _async_render_GET(self, request: SynapseRequest) -> None:
        await self._oidc_handler.handle_oidc_callback(request)

    async def _async_render_POST(self, request: SynapseRequest) -> None:
        await self._oidc_handler.handle_oidc_callback(request)