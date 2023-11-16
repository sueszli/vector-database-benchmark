"""MONGODB-AWS Authentication helpers."""
from __future__ import annotations
try:
    import pymongo_auth_aws
    from pymongo_auth_aws import AwsCredential, AwsSaslContext, PyMongoAuthAwsError
    _HAVE_MONGODB_AWS = True
except ImportError:

    class AwsSaslContext:

        def __init__(self, credentials: MongoCredential):
            if False:
                i = 10
                return i + 15
            pass
    _HAVE_MONGODB_AWS = False
try:
    from pymongo_auth_aws.auth import set_cached_credentials, set_use_cached_credentials
    set_use_cached_credentials(True)
except ImportError:

    def set_cached_credentials(_creds: Optional[AwsCredential]) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
if TYPE_CHECKING:
    from bson.typings import _ReadableBuffer
    from pymongo.auth import MongoCredential
    from pymongo.pool import Connection

class _AwsSaslContext(AwsSaslContext):

    def binary_type(self) -> Type[Binary]:
        if False:
            while True:
                i = 10
        'Return the bson.binary.Binary type.'
        return Binary

    def bson_encode(self, doc: Mapping[str, Any]) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        'Encode a dictionary to BSON.'
        return bson.encode(doc)

    def bson_decode(self, data: _ReadableBuffer) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        'Decode BSON to a dictionary.'
        return bson.decode(data)

def _authenticate_aws(credentials: MongoCredential, conn: Connection) -> None:
    if False:
        i = 10
        return i + 15
    'Authenticate using MONGODB-AWS.'
    if not _HAVE_MONGODB_AWS:
        raise ConfigurationError("MONGODB-AWS authentication requires pymongo-auth-aws: install with: python -m pip install 'pymongo[aws]'")
    if conn.max_wire_version < 9:
        raise ConfigurationError('MONGODB-AWS authentication requires MongoDB version 4.4 or later')
    try:
        ctx = _AwsSaslContext(AwsCredential(credentials.username, credentials.password, credentials.mechanism_properties.aws_session_token))
        client_payload = ctx.step(None)
        client_first = SON([('saslStart', 1), ('mechanism', 'MONGODB-AWS'), ('payload', client_payload)])
        server_first = conn.command('$external', client_first)
        res = server_first
        for _ in range(10):
            client_payload = ctx.step(res['payload'])
            cmd = SON([('saslContinue', 1), ('conversationId', server_first['conversationId']), ('payload', client_payload)])
            res = conn.command('$external', cmd)
            if res['done']:
                break
    except PyMongoAuthAwsError as exc:
        set_cached_credentials(None)
        raise OperationFailure(f'{exc} (pymongo-auth-aws version {pymongo_auth_aws.__version__})') from None
    except Exception:
        set_cached_credentials(None)
        raise