from __future__ import annotations
from typing import Any
from urllib.parse import urlencode
from django.core import signing
from django.urls import reverse
from sentry_sdk.api import capture_exception
from sentry import features, options
from sentry.models.organization import Organization
from sentry.services.hybrid_cloud.user.service import user_service
from sentry.types.region import get_local_region
from sentry.utils.numbers import base36_decode, base36_encode

def get_signer():
    if False:
        for i in range(10):
            print('nop')
    return signing.TimestampSigner(salt='sentry-link-signature')

def generate_signed_link(user, viewname: str, referrer: str | None=None, args: list[Any] | None=None, kwargs: dict[str, Any] | None=None):
    if False:
        i = 10
        return i + 15
    'This returns an absolute URL where the given user is signed in for\n    the given viewname with args and kwargs.  This returns a redirect link\n    that if followed sends the user to another URL which carries another\n    signature that is valid for that URL only.  The user can also be a user\n    ID.\n    '
    if hasattr(user, 'is_authenticated'):
        if not user.is_authenticated:
            raise RuntimeError('Need an authenticated user to sign a link.')
        user_id = user.id
    else:
        user_id = user
    path = reverse(viewname, args=args, kwargs=kwargs)
    item = '{}|{}|{}'.format(options.get('system.url-prefix'), path, base36_encode(user_id))
    signature = ':'.join(get_signer().sign(item).rsplit(':', 2)[1:])
    region = get_local_region()
    signed_link = f'{region.to_url(path)}?_={base36_encode(user_id)}:{signature}'
    if referrer:
        signed_link = signed_link + '&' + urlencode({'referrer': referrer})
    return signed_link

def generate_signed_unsubscribe_link(organization: Organization, user_id: int, resource: str, resource_id: str | int, referrer: str | None=None):
    if False:
        return 10
    '\n    Generate an absolute URL to the react rendered unsubscribe views\n\n    The URL will include a signature for the API endpoint that does read/writes.\n    The signature encodes the specific API path and userid that the action\n    is valid for.\n\n    The generated link will honour the customer-domain option for\n    the organization.\n    '
    html_viewname = f'sentry-organization-unsubscribe-{resource}'
    api_endpointname = f'sentry-api-0-organization-unsubscribe-{resource}'
    url_args = [organization.slug, resource_id]
    if features.has('organizations:customer-domains', organization):
        url_args = [resource_id]
        html_viewname = f'sentry-customer-domain-unsubscribe-{resource}'
    htmlpath = reverse(html_viewname, args=url_args)
    apipath = reverse(api_endpointname, args=[organization.slug, resource_id])
    item = '{}|{}|{}'.format(options.get('system.url-prefix'), apipath, base36_encode(user_id))
    signature = ':'.join(get_signer().sign(item).rsplit(':', 2)[1:])
    query = f'_={base36_encode(user_id)}:{signature}'
    if referrer:
        query = query + '&' + urlencode({'referrer': referrer})
    return organization.absolute_url(path=htmlpath, query=query)

def find_signature(request) -> str | None:
    if False:
        while True:
            i = 10
    return request.GET.get('_')

def process_signature(request, max_age=60 * 60 * 24 * 10):
    if False:
        return 10
    'Given a request object this validates the signature from the\n    current request and returns the user.\n    '
    sig = find_signature(request)
    if not sig or sig.count(':') < 2:
        return None
    url_prefix = options.get('system.url-prefix')
    request_path = request.path
    signed_data = f'{url_prefix}|{request_path}|{sig}'
    try:
        data = get_signer().unsign(signed_data, max_age=max_age)
    except signing.BadSignature as e:
        capture_exception(e)
        return None
    (_, signed_path, user_id) = data.rsplit('|', 2)
    if signed_path != request.path:
        return None
    try:
        return user_service.get_user(user_id=base36_decode(user_id))
    except ValueError:
        return None