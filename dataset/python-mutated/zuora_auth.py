from typing import Any, Dict, Mapping
from airbyte_cdk.sources.streams.http.requests_native_auth.oauth import Oauth2Authenticator
from .zuora_endpoint import get_url_base

class OAuth(Oauth2Authenticator):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)

    def get_refresh_request_body(self) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        payload = super().get_refresh_request_body()
        payload['grant_type'] = 'client_credentials'
        payload.pop('refresh_token')
        return payload

class ZuoraAuthenticator:

    def __init__(self, config: Dict):
        if False:
            i = 10
            return i + 15
        self.config = config

    @property
    def url_base(self) -> str:
        if False:
            while True:
                i = 10
        return get_url_base(self.config['tenant_endpoint'])

    def get_auth(self) -> OAuth:
        if False:
            for i in range(10):
                print('nop')
        return OAuth(token_refresh_endpoint=f'{self.url_base}/oauth/token', client_id=self.config['client_id'], client_secret=self.config['client_secret'], refresh_token=None)