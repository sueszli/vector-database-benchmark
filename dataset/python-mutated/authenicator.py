import logging
from typing import Any, Mapping
from urllib import parse
from airbyte_cdk.sources.streams.http.requests_native_auth import Oauth2Authenticator
logger = logging.getLogger('airbyte')

class GenesysOAuthAuthenticator(Oauth2Authenticator):

    def __init__(self, base_url: str, client_id: str, client_secret: str):
        if False:
            while True:
                i = 10
        super().__init__(parse.urljoin(base_url, '/oauth/token'), client_id, client_secret, '')

    def build_refresh_request_body(self) -> Mapping[str, Any]:
        if False:
            return 10
        if not self.get_refresh_token():
            return {'grant_type': 'client_credentials', 'client_id': self.get_client_id(), 'client_secret': self.get_client_secret()}
        else:
            return super().build_refresh_request_body()