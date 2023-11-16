import re
from typing import TYPE_CHECKING
from synapse.http.server import respond_with_json
from synapse.http.servlet import RestServlet
from synapse.http.site import SynapseRequest
if TYPE_CHECKING:
    from synapse.server import HomeServer

class MediaConfigResource(RestServlet):
    PATTERNS = [re.compile('/_matrix/media/(r0|v3|v1)/config$')]

    def __init__(self, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        super().__init__()
        config = hs.config
        self.clock = hs.get_clock()
        self.auth = hs.get_auth()
        self.limits_dict = {'m.upload.size': config.media.max_upload_size}

    async def on_GET(self, request: SynapseRequest) -> None:
        await self.auth.get_user_by_req(request)
        respond_with_json(request, 200, self.limits_dict, send_cors=True)