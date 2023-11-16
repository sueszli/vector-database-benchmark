import re
import urllib
from typing import Optional
from django.conf import settings
from django.http import HttpRequest
from zerver.lib.upload import get_public_upload_root_url
from zerver.models import Realm, UserProfile

def get_subdomain(request: HttpRequest) -> str:
    if False:
        while True:
            i = 10
    host = request.get_host().lower()
    return get_subdomain_from_hostname(host)

def get_subdomain_from_hostname(host: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    m = re.search(f'\\.{settings.EXTERNAL_HOST}(:\\d+)?$', host)
    if m:
        subdomain = host[:m.start()]
        if subdomain in settings.ROOT_SUBDOMAIN_ALIASES:
            return Realm.SUBDOMAIN_FOR_ROOT_DOMAIN
        return subdomain
    for (subdomain, realm_host) in settings.REALM_HOSTS.items():
        if re.search(f'^{realm_host}(:\\d+)?$', host):
            return subdomain
    return Realm.SUBDOMAIN_FOR_ROOT_DOMAIN

def is_subdomain_root_or_alias(request: HttpRequest) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return get_subdomain(request) == Realm.SUBDOMAIN_FOR_ROOT_DOMAIN

def user_matches_subdomain(realm_subdomain: str, user_profile: UserProfile) -> bool:
    if False:
        print('Hello World!')
    return user_profile.realm.subdomain == realm_subdomain

def is_root_domain_available() -> bool:
    if False:
        for i in range(10):
            print('nop')
    if settings.ROOT_DOMAIN_LANDING_PAGE:
        return False
    return not Realm.objects.filter(string_id=Realm.SUBDOMAIN_FOR_ROOT_DOMAIN).exists()

def is_static_or_current_realm_url(url: str, realm: Optional[Realm]) -> bool:
    if False:
        i = 10
        return i + 15
    assert settings.STATIC_URL is not None
    split_url = urllib.parse.urlsplit(url)
    split_static_url = urllib.parse.urlsplit(settings.STATIC_URL)
    if split_url.netloc == split_static_url.netloc and url.startswith(settings.STATIC_URL):
        return True
    if realm is not None and split_url.netloc == realm.host and (f'{split_url.scheme}://' == settings.EXTERNAL_URI_SCHEME):
        return True
    if split_url.netloc == '' and split_url.scheme == '':
        return True
    if settings.LOCAL_UPLOADS_DIR is None:
        public_upload_root_url = get_public_upload_root_url()
        assert public_upload_root_url.endswith('/')
        if url.startswith(public_upload_root_url):
            return True
    return False