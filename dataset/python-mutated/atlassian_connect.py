from __future__ import annotations
import hashlib
from typing import Mapping, Sequence
import requests
from django.http import HttpRequest
from jwt import InvalidSignatureError
from rest_framework.request import Request
from sentry.models.integrations.integration import Integration
from sentry.services.hybrid_cloud.integration.model import RpcIntegration
from sentry.services.hybrid_cloud.integration.service import integration_service
from sentry.services.hybrid_cloud.util import control_silo_function
from sentry.utils import jwt
from sentry.utils.http import absolute_uri, percent_encode

class AtlassianConnectValidationError(Exception):
    pass

def get_query_hash(uri: str, method: str, query_params: Mapping[str, str | Sequence[str]] | None=None) -> str:
    if False:
        while True:
            i = 10
    uri = uri.rstrip('/')
    method = method.upper()
    if query_params is None:
        query_params = {}
    sorted_query = []
    for (k, v) in sorted(query_params.items()):
        if k != 'jwt':
            if isinstance(v, str):
                param_val = percent_encode(v)
            else:
                param_val = ','.join((percent_encode(val) for val in v))
            sorted_query.append(f'{percent_encode(k)}={param_val}')
    query_string = '{}&{}&{}'.format(method, uri, '&'.join(sorted_query))
    return hashlib.sha256(query_string.encode('utf8')).hexdigest()

def get_token(request: HttpRequest) -> str:
    if False:
        for i in range(10):
            print('nop')
    try:
        auth_header: str = request.META['HTTP_AUTHORIZATION']
        return auth_header.split(' ', 1)[1]
    except (KeyError, IndexError):
        raise AtlassianConnectValidationError('Missing/Invalid authorization header')

def get_integration_from_jwt(token: str | None, path: str, provider: str, query_params: Mapping[str, str] | None, method: str='GET') -> RpcIntegration:
    if False:
        print('Hello World!')
    if token is None:
        raise AtlassianConnectValidationError('No token parameter')
    claims = jwt.peek_claims(token)
    headers = jwt.peek_header(token)
    issuer = claims.get('iss')
    integration = integration_service.get_integration(provider=provider, external_id=issuer)
    if not integration:
        raise AtlassianConnectValidationError('No integration found')
    key_id = headers.get('kid')
    try:
        decoded_claims = authenticate_asymmetric_jwt(token, key_id) if key_id else jwt.decode(token, integration.metadata['shared_secret'], audience=False)
    except InvalidSignatureError:
        raise AtlassianConnectValidationError('Signature is invalid')
    verify_claims(decoded_claims, path, query_params, method)
    return integration

def verify_claims(claims: Mapping[str, str], path: str, query_params: Mapping[str, str] | None, method: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    qsh = get_query_hash(path, method, query_params)
    if qsh != claims['qsh']:
        raise AtlassianConnectValidationError('Query hash mismatch')

def authenticate_asymmetric_jwt(token: str | None, key_id: str) -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    '\n    Allows for Atlassian Connect installation lifecycle security improvements (i.e. verified senders)\n    See: https://community.developer.atlassian.com/t/action-required-atlassian-connect-installation-lifecycle-security-improvements/49046\n    '
    if token is None:
        raise AtlassianConnectValidationError('No token parameter')
    headers = jwt.peek_header(token)
    key_response = requests.get(f'https://connect-install-keys.atlassian.com/{key_id}')
    public_key = key_response.content.decode('utf-8').strip()
    decoded_claims = jwt.decode(token, public_key, audience=absolute_uri(), algorithms=[headers.get('alg')])
    if not decoded_claims:
        raise AtlassianConnectValidationError('Unable to verify asymmetric installation JWT')
    return decoded_claims

def get_integration_from_request(request: Request, provider: str) -> RpcIntegration:
    if False:
        return 10
    return get_integration_from_jwt(request.GET.get('jwt'), request.path, provider, request.GET)

@control_silo_function
def parse_integration_from_request(request: HttpRequest, provider: str) -> Integration | None:
    if False:
        return 10
    token = get_token(request=request) if request.META.get('HTTP_AUTHORIZATION') is not None else request.GET.get('jwt')
    rpc_integration = get_integration_from_jwt(token=token, path=request.path, provider=provider, query_params=request.GET, method=request.method if request.method else 'POST')
    return Integration.objects.filter(id=rpc_integration.id).first()