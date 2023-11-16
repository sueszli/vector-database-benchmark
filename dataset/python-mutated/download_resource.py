import logging
import re
from typing import TYPE_CHECKING, Optional
from synapse.http.server import set_corp_headers, set_cors_headers
from synapse.http.servlet import RestServlet, parse_boolean, parse_integer
from synapse.http.site import SynapseRequest
from synapse.media._base import DEFAULT_MAX_TIMEOUT_MS, MAXIMUM_ALLOWED_MAX_TIMEOUT_MS, respond_404
from synapse.util.stringutils import parse_and_validate_server_name
if TYPE_CHECKING:
    from synapse.media.media_repository import MediaRepository
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class DownloadResource(RestServlet):
    PATTERNS = [re.compile('/_matrix/media/(r0|v3|v1)/download/(?P<server_name>[^/]*)/(?P<media_id>[^/]*)(/(?P<file_name>[^/]*))?$')]

    def __init__(self, hs: 'HomeServer', media_repo: 'MediaRepository'):
        if False:
            return 10
        super().__init__()
        self.media_repo = media_repo
        self._is_mine_server_name = hs.is_mine_server_name

    async def on_GET(self, request: SynapseRequest, server_name: str, media_id: str, file_name: Optional[str]=None) -> None:
        parse_and_validate_server_name(server_name)
        set_cors_headers(request)
        set_corp_headers(request)
        request.setHeader(b'Content-Security-Policy', b"sandbox; default-src 'none'; script-src 'none'; plugin-types application/pdf; style-src 'unsafe-inline'; media-src 'self'; object-src 'self';")
        request.setHeader(b'X-Content-Security-Policy', b'sandbox;')
        request.setHeader(b'Referrer-Policy', b'no-referrer')
        max_timeout_ms = parse_integer(request, 'timeout_ms', default=DEFAULT_MAX_TIMEOUT_MS)
        max_timeout_ms = min(max_timeout_ms, MAXIMUM_ALLOWED_MAX_TIMEOUT_MS)
        if self._is_mine_server_name(server_name):
            await self.media_repo.get_local_media(request, media_id, file_name, max_timeout_ms)
        else:
            allow_remote = parse_boolean(request, 'allow_remote', default=True)
            if not allow_remote:
                logger.info('Rejecting request for remote media %s/%s due to allow_remote', server_name, media_id)
                respond_404(request)
                return
            await self.media_repo.get_remote_media(request, server_name, media_id, file_name, max_timeout_ms)