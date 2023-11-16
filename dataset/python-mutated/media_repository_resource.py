from typing import TYPE_CHECKING
from synapse.config._base import ConfigError
from synapse.http.server import HttpServer, JsonResource
from .config_resource import MediaConfigResource
from .create_resource import CreateResource
from .download_resource import DownloadResource
from .preview_url_resource import PreviewUrlResource
from .thumbnail_resource import ThumbnailResource
from .upload_resource import AsyncUploadServlet, UploadServlet
if TYPE_CHECKING:
    from synapse.server import HomeServer

class MediaRepositoryResource(JsonResource):
    """File uploading and downloading.

    Uploads are POSTed to a resource which returns a token which is used to GET
    the download::

        => POST /_matrix/media/r0/upload HTTP/1.1
           Content-Type: <media-type>
           Content-Length: <content-length>

           <media>

        <= HTTP/1.1 200 OK
           Content-Type: application/json

           { "content_uri": "mxc://<server-name>/<media-id>" }

        => GET /_matrix/media/r0/download/<server-name>/<media-id> HTTP/1.1

        <= HTTP/1.1 200 OK
           Content-Type: <media-type>
           Content-Disposition: attachment;filename=<upload-filename>

           <media>

    Clients can get thumbnails by supplying a desired width and height and
    thumbnailing method::

        => GET /_matrix/media/r0/thumbnail/<server_name>
                /<media-id>?width=<w>&height=<h>&method=<m> HTTP/1.1

        <= HTTP/1.1 200 OK
           Content-Type: image/jpeg or image/png

           <thumbnail>

    The thumbnail methods are "crop" and "scale". "scale" tries to return an
    image where either the width or the height is smaller than the requested
    size. The client should then scale and letterbox the image if it needs to
    fit within a given rectangle. "crop" tries to return an image where the
    width and height are close to the requested size and the aspect matches
    the requested size. The client should scale the image if it needs to fit
    within a given rectangle.

    This gets mounted at various points under /_matrix/media, including:
       * /_matrix/media/r0
       * /_matrix/media/v1
       * /_matrix/media/v3
    """

    def __init__(self, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        if not hs.config.media.can_load_media_repo:
            raise ConfigError('Synapse is not configured to use a media repo.')
        JsonResource.__init__(self, hs, canonical_json=False)
        self.register_servlets(self, hs)

    @staticmethod
    def register_servlets(http_server: HttpServer, hs: 'HomeServer') -> None:
        if False:
            for i in range(10):
                print('nop')
        media_repo = hs.get_media_repository()
        CreateResource(hs, media_repo).register(http_server)
        UploadServlet(hs, media_repo).register(http_server)
        AsyncUploadServlet(hs, media_repo).register(http_server)
        DownloadResource(hs, media_repo).register(http_server)
        ThumbnailResource(hs, media_repo, media_repo.media_storage).register(http_server)
        if hs.config.media.url_preview_enabled:
            PreviewUrlResource(hs, media_repo, media_repo.media_storage).register(http_server)
        MediaConfigResource(hs).register(http_server)