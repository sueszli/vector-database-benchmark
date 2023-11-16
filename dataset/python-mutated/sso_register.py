import logging
from typing import TYPE_CHECKING
from twisted.web.server import Request
from synapse.api.errors import SynapseError
from synapse.handlers.sso import get_username_mapping_session_cookie_from_request
from synapse.http.server import DirectServeHtmlResource
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class SsoRegisterResource(DirectServeHtmlResource):
    """A resource which completes SSO registration

    This resource gets mounted at /_synapse/client/sso_register, and is shown
    after we collect username and/or consent for a new SSO user. It (finally) registers
    the user, and confirms redirect to the client
    """

    def __init__(self, hs: 'HomeServer'):
        if False:
            return 10
        super().__init__()
        self._sso_handler = hs.get_sso_handler()

    async def _async_render_GET(self, request: Request) -> None:
        try:
            session_id = get_username_mapping_session_cookie_from_request(request)
        except SynapseError as e:
            logger.warning('Error fetching session cookie: %s', e)
            self._sso_handler.render_error(request, 'bad_session', e.msg, code=e.code)
            return
        await self._sso_handler.register_sso_user(request, session_id)