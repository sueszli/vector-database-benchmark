from typing import TYPE_CHECKING
from twisted.web.server import Request
from synapse.http.server import DirectServeHtmlResource
from synapse.http.site import SynapseRequest
if TYPE_CHECKING:
    from synapse.server import HomeServer

class SAML2ResponseResource(DirectServeHtmlResource):
    """A Twisted web resource which handles the SAML response"""
    isLeaf = 1

    def __init__(self, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        super().__init__()
        self._saml_handler = hs.get_saml_handler()
        self._sso_handler = hs.get_sso_handler()

    async def _async_render_GET(self, request: Request) -> None:
        self._sso_handler.render_error(request, 'unexpected_get', 'Unexpected GET request on /saml2/authn_response')

    async def _async_render_POST(self, request: SynapseRequest) -> None:
        await self._saml_handler.handle_saml_response(request)