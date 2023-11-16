import logging
import re
from typing import TYPE_CHECKING, Optional, Tuple
from signedjson.sign import sign_json
from unpaddedbase64 import encode_base64
from twisted.web.server import Request
from synapse.http.servlet import RestServlet
from synapse.types import JsonDict
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class LocalKey(RestServlet):
    """HTTP resource containing encoding the TLS X.509 certificate and NACL
    signature verification keys for this server::

        GET /_matrix/key/v2/server HTTP/1.1

        GET /_matrix/key/v2/server/a.key.id HTTP/1.1

        HTTP/1.1 200 OK
        Content-Type: application/json
        {
            "valid_until_ts": # integer posix timestamp when this result expires.
            "server_name": "this.server.example.com"
            "verify_keys": {
                "algorithm:version": {
                    "key": # base64 encoded NACL verification key.
                }
            },
            "old_verify_keys": {
                "algorithm:version": {
                    "expired_ts": # integer posix timestamp when the key expired.
                    "key": # base64 encoded NACL verification key.
                }
            },
            "signatures": {
                "this.server.example.com": {
                   "algorithm:version": # NACL signature for this server
                }
            }
        }
    """
    PATTERNS = (re.compile('^/_matrix/key/v2/server(/(?P<key_id>[^/]*))?$'),)

    def __init__(self, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        self.config = hs.config
        self.clock = hs.get_clock()
        self.update_response_body(self.clock.time_msec())

    def update_response_body(self, time_now_msec: int) -> None:
        if False:
            i = 10
            return i + 15
        refresh_interval = self.config.key.key_refresh_interval
        self.valid_until_ts = int(time_now_msec + refresh_interval)
        self.response_body = self.response_json_object()

    def response_json_object(self) -> JsonDict:
        if False:
            for i in range(10):
                print('nop')
        verify_keys = {}
        for signing_key in self.config.key.signing_key:
            verify_key_bytes = signing_key.verify_key.encode()
            key_id = '%s:%s' % (signing_key.alg, signing_key.version)
            verify_keys[key_id] = {'key': encode_base64(verify_key_bytes)}
        old_verify_keys = {}
        for (key_id, old_signing_key) in self.config.key.old_signing_keys.items():
            verify_key_bytes = old_signing_key.encode()
            old_verify_keys[key_id] = {'key': encode_base64(verify_key_bytes), 'expired_ts': old_signing_key.expired}
        json_object = {'valid_until_ts': self.valid_until_ts, 'server_name': self.config.server.server_name, 'verify_keys': verify_keys, 'old_verify_keys': old_verify_keys}
        for key in self.config.key.signing_key:
            json_object = sign_json(json_object, self.config.server.server_name, key)
        return json_object

    def on_GET(self, request: Request, key_id: Optional[str]=None) -> Tuple[int, JsonDict]:
        if False:
            print('Hello World!')
        if key_id:
            logger.warning('Request for local server key with deprecated key ID (logging to determine usage level for future removal): %s', key_id)
        time_now = self.clock.time_msec()
        if time_now + self.config.key.key_refresh_interval / 2 > self.valid_until_ts:
            self.update_response_body(time_now)
        return (200, self.response_body)