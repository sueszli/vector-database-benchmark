import logging
from typing import TYPE_CHECKING, Tuple
from synapse.http.server import DirectServeJsonResource
from synapse.http.site import SynapseRequest
from synapse.types import JsonDict
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class JwksResource(DirectServeJsonResource):

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        super().__init__(extract_context=True)
        public_parameters = {'kty', 'use', 'key_ops', 'alg', 'kid', 'x5u', 'x5c', 'x5t', 'x5t#S256', 'crv', 'x', 'y', 'n', 'e', 'ext'}
        key = hs.config.experimental.msc3861.jwk
        if key is not None:
            private_key = key.as_dict()
            public_key = {k: v for (k, v) in private_key.items() if k in public_parameters}
            keys = [public_key]
        else:
            keys = []
        self.res = {'keys': keys}

    async def _async_render_GET(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        return (200, self.res)