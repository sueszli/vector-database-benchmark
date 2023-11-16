import logging
import re
from typing import IO, TYPE_CHECKING, Dict, List, Optional, Tuple
from synapse.api.errors import Codes, SynapseError
from synapse.http.server import respond_with_json
from synapse.http.servlet import RestServlet, parse_bytes_from_args
from synapse.http.site import SynapseRequest
from synapse.media.media_storage import SpamMediaException
if TYPE_CHECKING:
    from synapse.media.media_repository import MediaRepository
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
_UPLOAD_MEDIA_LOCK_NAME = 'upload_media'

class BaseUploadServlet(RestServlet):

    def __init__(self, hs: 'HomeServer', media_repo: 'MediaRepository'):
        if False:
            while True:
                i = 10
        super().__init__()
        self.media_repo = media_repo
        self.filepaths = media_repo.filepaths
        self.store = hs.get_datastores().main
        self.server_name = hs.hostname
        self.auth = hs.get_auth()
        self.max_upload_size = hs.config.media.max_upload_size

    def _get_file_metadata(self, request: SynapseRequest) -> Tuple[int, Optional[str], str]:
        if False:
            return 10
        raw_content_length = request.getHeader('Content-Length')
        if raw_content_length is None:
            raise SynapseError(msg='Request must specify a Content-Length', code=400)
        try:
            content_length = int(raw_content_length)
        except ValueError:
            raise SynapseError(msg='Content-Length value is invalid', code=400)
        if content_length > self.max_upload_size:
            raise SynapseError(msg='Upload request body is too large', code=413, errcode=Codes.TOO_LARGE)
        args: Dict[bytes, List[bytes]] = request.args
        upload_name_bytes = parse_bytes_from_args(args, 'filename')
        if upload_name_bytes:
            try:
                upload_name: Optional[str] = upload_name_bytes.decode('utf8')
            except UnicodeDecodeError:
                raise SynapseError(msg='Invalid UTF-8 filename parameter: %r' % (upload_name_bytes,), code=400)
        else:
            upload_name = None
        headers = request.requestHeaders
        if headers.hasHeader(b'Content-Type'):
            content_type_headers = headers.getRawHeaders(b'Content-Type')
            assert content_type_headers
            media_type = content_type_headers[0].decode('ascii')
        else:
            media_type = 'application/octet-stream'
        return (content_length, upload_name, media_type)

class UploadServlet(BaseUploadServlet):
    PATTERNS = [re.compile('/_matrix/media/(r0|v3|v1)/upload$')]

    async def on_POST(self, request: SynapseRequest) -> None:
        requester = await self.auth.get_user_by_req(request)
        (content_length, upload_name, media_type) = self._get_file_metadata(request)
        try:
            content: IO = request.content
            content_uri = await self.media_repo.create_content(media_type, upload_name, content, content_length, requester.user)
        except SpamMediaException:
            raise SynapseError(400, 'Bad content')
        logger.info("Uploaded content with URI '%s'", content_uri)
        respond_with_json(request, 200, {'content_uri': str(content_uri)}, send_cors=True)

class AsyncUploadServlet(BaseUploadServlet):
    PATTERNS = [re.compile('/_matrix/media/v3/upload/(?P<server_name>[^/]*)/(?P<media_id>[^/]*)$')]

    async def on_PUT(self, request: SynapseRequest, server_name: str, media_id: str) -> None:
        requester = await self.auth.get_user_by_req(request)
        if server_name != self.server_name:
            raise SynapseError(404, 'Non-local server name specified', errcode=Codes.NOT_FOUND)
        lock = await self.store.try_acquire_lock(_UPLOAD_MEDIA_LOCK_NAME, media_id)
        if not lock:
            raise SynapseError(409, 'Media ID cannot be overwritten', errcode=Codes.CANNOT_OVERWRITE_MEDIA)
        async with lock:
            await self.media_repo.verify_can_upload(media_id, requester.user)
            (content_length, upload_name, media_type) = self._get_file_metadata(request)
            try:
                content: IO = request.content
                await self.media_repo.update_content(media_id, media_type, upload_name, content, content_length, requester.user)
            except SpamMediaException:
                raise SynapseError(400, 'Bad content')
            logger.info('Uploaded content for media ID %r', media_id)
            respond_with_json(request, 200, {}, send_cors=True)