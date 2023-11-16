import logging
from typing import TYPE_CHECKING
from synapse.http.server import DirectServeHtmlResource, finish_request, respond_with_html
from synapse.http.servlet import parse_string
from synapse.http.site import SynapseRequest
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class PickIdpResource(DirectServeHtmlResource):
    """IdP picker resource.

    This resource gets mounted under /_synapse/client/pick_idp. It serves an HTML page
    which prompts the user to choose an Identity Provider from the list.
    """

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._sso_handler = hs.get_sso_handler()
        self._sso_login_idp_picker_template = hs.config.sso.sso_login_idp_picker_template
        self._server_name = hs.hostname

    async def _async_render_GET(self, request: SynapseRequest) -> None:
        client_redirect_url = parse_string(request, 'redirectUrl', required=True, encoding='utf-8')
        idp = parse_string(request, 'idp', required=False)
        if not idp:
            return await self._serve_id_picker(request, client_redirect_url)
        providers = self._sso_handler.get_identity_providers()
        auth_provider = providers.get(idp)
        if not auth_provider:
            logger.info('Unknown idp %r', idp)
            self._sso_handler.render_error(request, 'unknown_idp', 'Unknown identity provider ID')
            return
        sso_url = await auth_provider.handle_redirect_request(request, client_redirect_url.encode('utf8'))
        logger.info('Redirecting to %s', sso_url)
        request.redirect(sso_url)
        finish_request(request)

    async def _serve_id_picker(self, request: SynapseRequest, client_redirect_url: str) -> None:
        providers = self._sso_handler.get_identity_providers()
        html = self._sso_login_idp_picker_template.render(redirect_url=client_redirect_url, server_name=self._server_name, providers=providers.values())
        respond_with_html(request, 200, html)