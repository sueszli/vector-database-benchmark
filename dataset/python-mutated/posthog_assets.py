import re
from typing import List
from django.conf import settings
from django.template import Library
from posthog.utils import absolute_uri as util_absolute_uri
register = Library()

@register.simple_tag
def absolute_uri(url: str='') -> str:
    if False:
        for i in range(10):
            print('nop')
    return util_absolute_uri(url)

@register.simple_tag
def absolute_asset_url(path: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a versioned absolute asset URL (located within PostHog\'s static files).\n    Example:\n      {% absolute_asset_url \'dist/posthog.css\' %}\n      =>  "http://posthog.example.com/_static/74d127b78dc7daf2c51f/dist/posthog.css"\n    '
    return absolute_uri(f"{settings.STATIC_URL.rstrip('/')}/{path.lstrip('/')}")

@register.simple_tag
def human_social_providers(providers: List[str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a human-friendly name for a social login provider.\n    Example:\n      {% human_social_providers ["google-oauth2", "github"] %}\n      =>  "Google, GitHub"\n    '

    def friendly_provider(prov: str) -> str:
        if False:
            print('Hello World!')
        if prov == 'google-oauth2':
            return 'Google'
        elif prov == 'github':
            return 'GitHub'
        elif prov == 'gitlab':
            return 'GitLab'
        return 'single sign-on (SAML)'
    return ', '.join(map(friendly_provider, providers))

@register.simple_tag
def strip_protocol(path: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Returns a URL removing the http/https protocol\n    Example:\n      {% strip_protocol \'https://app.posthog.com\' %}\n      =>  "app.posthog.com"\n    '
    return re.sub('https?:\\/\\/', '', path)