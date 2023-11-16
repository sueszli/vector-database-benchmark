import logging
import string
from http import HTTPStatus
from typing import TYPE_CHECKING, Tuple
from synapse.api.errors import Codes, NotFoundError, SynapseError
from synapse.http.servlet import RestServlet, parse_boolean, parse_json_object_from_request
from synapse.http.site import SynapseRequest
from synapse.rest.admin._base import admin_patterns, assert_requester_is_admin
from synapse.types import JsonDict
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class ListRegistrationTokensRestServlet(RestServlet):
    """List registration tokens.

    To list all tokens:

        GET /_synapse/admin/v1/registration_tokens

        200 OK

        {
            "registration_tokens": [
                {
                    "token": "abcd",
                    "uses_allowed": 3,
                    "pending": 0,
                    "completed": 1,
                    "expiry_time": null
                },
                {
                    "token": "wxyz",
                    "uses_allowed": null,
                    "pending": 0,
                    "completed": 9,
                    "expiry_time": 1625394937000
                }
            ]
        }

    The optional query parameter `valid` can be used to filter the response.
    If it is `true`, only valid tokens are returned. If it is `false`, only
    tokens that have expired or have had all uses exhausted are returned.
    If it is omitted, all tokens are returned regardless of validity.
    """
    PATTERNS = admin_patterns('/registration_tokens$')

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        self.auth = hs.get_auth()
        self.store = hs.get_datastores().main

    async def on_GET(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        await assert_requester_is_admin(self.auth, request)
        valid = parse_boolean(request, 'valid')
        token_list = await self.store.get_registration_tokens(valid)
        return (HTTPStatus.OK, {'registration_tokens': [{'token': t[0], 'uses_allowed': t[1], 'pending': t[2], 'completed': t[3], 'expiry_time': t[4]} for t in token_list]})

class NewRegistrationTokenRestServlet(RestServlet):
    """Create a new registration token.

    For example, to create a token specifying some fields:

        POST /_synapse/admin/v1/registration_tokens/new

        {
            "token": "defg",
            "uses_allowed": 1
        }

        200 OK

        {
            "token": "defg",
            "uses_allowed": 1,
            "pending": 0,
            "completed": 0,
            "expiry_time": null
        }

    Defaults are used for any fields not specified.
    """
    PATTERNS = admin_patterns('/registration_tokens/new$')

    def __init__(self, hs: 'HomeServer'):
        if False:
            return 10
        self.auth = hs.get_auth()
        self.store = hs.get_datastores().main
        self.clock = hs.get_clock()
        self.allowed_chars = string.ascii_letters + string.digits + '._~-'
        self.allowed_chars_set = set(self.allowed_chars)

    async def on_POST(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        await assert_requester_is_admin(self.auth, request)
        body = parse_json_object_from_request(request)
        if 'token' in body:
            token = body['token']
            if not isinstance(token, str):
                raise SynapseError(HTTPStatus.BAD_REQUEST, 'token must be a string', Codes.INVALID_PARAM)
            if not 0 < len(token) <= 64:
                raise SynapseError(HTTPStatus.BAD_REQUEST, 'token must not be empty and must not be longer than 64 characters', Codes.INVALID_PARAM)
            if not set(token).issubset(self.allowed_chars_set):
                raise SynapseError(HTTPStatus.BAD_REQUEST, 'token must consist only of characters matched by the regex [A-Za-z0-9-_]', Codes.INVALID_PARAM)
        else:
            length = body.get('length', 16)
            if type(length) is not int:
                raise SynapseError(HTTPStatus.BAD_REQUEST, 'length must be an integer', Codes.INVALID_PARAM)
            if not 0 < length <= 64:
                raise SynapseError(HTTPStatus.BAD_REQUEST, 'length must be greater than zero and not greater than 64', Codes.INVALID_PARAM)
            token = await self.store.generate_registration_token(length, self.allowed_chars)
        uses_allowed = body.get('uses_allowed', None)
        if not (uses_allowed is None or (type(uses_allowed) is int and uses_allowed >= 0)):
            raise SynapseError(HTTPStatus.BAD_REQUEST, 'uses_allowed must be a non-negative integer or null', Codes.INVALID_PARAM)
        expiry_time = body.get('expiry_time', None)
        if expiry_time is not None and type(expiry_time) is not int:
            raise SynapseError(HTTPStatus.BAD_REQUEST, 'expiry_time must be an integer or null', Codes.INVALID_PARAM)
        if type(expiry_time) is int and expiry_time < self.clock.time_msec():
            raise SynapseError(HTTPStatus.BAD_REQUEST, 'expiry_time must not be in the past', Codes.INVALID_PARAM)
        created = await self.store.create_registration_token(token, uses_allowed, expiry_time)
        if not created:
            raise SynapseError(HTTPStatus.BAD_REQUEST, f'Token already exists: {token}', Codes.INVALID_PARAM)
        resp = {'token': token, 'uses_allowed': uses_allowed, 'pending': 0, 'completed': 0, 'expiry_time': expiry_time}
        return (HTTPStatus.OK, resp)

class RegistrationTokenRestServlet(RestServlet):
    """Retrieve, update, or delete the given token.

    For example,

    to retrieve a token:

        GET /_synapse/admin/v1/registration_tokens/abcd

        200 OK

        {
            "token": "abcd",
            "uses_allowed": 3,
            "pending": 0,
            "completed": 1,
            "expiry_time": null
        }


    to update a token:

        PUT /_synapse/admin/v1/registration_tokens/defg

        {
            "uses_allowed": 5,
            "expiry_time": 4781243146000
        }

        200 OK

        {
            "token": "defg",
            "uses_allowed": 5,
            "pending": 0,
            "completed": 0,
            "expiry_time": 4781243146000
        }


    to delete a token:

        DELETE /_synapse/admin/v1/registration_tokens/wxyz

        200 OK

        {}
    """
    PATTERNS = admin_patterns('/registration_tokens/(?P<token>[^/]*)$')

    def __init__(self, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        self.clock = hs.get_clock()
        self.auth = hs.get_auth()
        self.store = hs.get_datastores().main

    async def on_GET(self, request: SynapseRequest, token: str) -> Tuple[int, JsonDict]:
        """Retrieve a registration token."""
        await assert_requester_is_admin(self.auth, request)
        token_info = await self.store.get_one_registration_token(token)
        if token_info is None:
            raise NotFoundError(f'No such registration token: {token}')
        return (HTTPStatus.OK, token_info)

    async def on_PUT(self, request: SynapseRequest, token: str) -> Tuple[int, JsonDict]:
        """Update a registration token."""
        await assert_requester_is_admin(self.auth, request)
        body = parse_json_object_from_request(request)
        new_attributes = {}
        if 'uses_allowed' in body:
            uses_allowed = body['uses_allowed']
            if not (uses_allowed is None or (type(uses_allowed) is int and uses_allowed >= 0)):
                raise SynapseError(HTTPStatus.BAD_REQUEST, 'uses_allowed must be a non-negative integer or null', Codes.INVALID_PARAM)
            new_attributes['uses_allowed'] = uses_allowed
        if 'expiry_time' in body:
            expiry_time = body['expiry_time']
            if expiry_time is not None and type(expiry_time) is not int:
                raise SynapseError(HTTPStatus.BAD_REQUEST, 'expiry_time must be an integer or null', Codes.INVALID_PARAM)
            if type(expiry_time) is int and expiry_time < self.clock.time_msec():
                raise SynapseError(HTTPStatus.BAD_REQUEST, 'expiry_time must not be in the past', Codes.INVALID_PARAM)
            new_attributes['expiry_time'] = expiry_time
        if len(new_attributes) == 0:
            token_info = await self.store.get_one_registration_token(token)
        else:
            token_info = await self.store.update_registration_token(token, new_attributes)
        if token_info is None:
            raise NotFoundError(f'No such registration token: {token}')
        return (HTTPStatus.OK, token_info)

    async def on_DELETE(self, request: SynapseRequest, token: str) -> Tuple[int, JsonDict]:
        """Delete a registration token."""
        await assert_requester_is_admin(self.auth, request)
        if await self.store.delete_registration_token(token):
            return (HTTPStatus.OK, {})
        raise NotFoundError(f'No such registration token: {token}')