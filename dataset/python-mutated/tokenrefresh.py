from typing import TYPE_CHECKING
from twisted.web.server import Request
from synapse.api.errors import AuthError
from synapse.http.server import HttpServer
from synapse.http.servlet import RestServlet
from ._base import client_patterns
if TYPE_CHECKING:
    from synapse.server import HomeServer

class TokenRefreshRestServlet(RestServlet):
    """
    Exchanges refresh tokens for a pair of an access token and a new refresh
    token.
    """
    PATTERNS = client_patterns('/tokenrefresh')

    def __init__(self, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        super().__init__()

    async def on_POST(self, request: Request) -> None:
        raise AuthError(403, 'tokenrefresh is no longer supported.')

def register_servlets(hs: 'HomeServer', http_server: HttpServer) -> None:
    if False:
        while True:
            i = 10
    TokenRefreshRestServlet(hs).register(http_server)