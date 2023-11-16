import dataclasses
import datetime
import enum
import io
import logging
import pathlib
import time
from collections import namedtuple
from http import HTTPStatus
from typing import Any, Callable, Dict, Generator, List, Mapping, NewType, Optional, Tuple, Union
import sentry_sdk
from dateutil.parser import parse as parse_date
from requests import Session, Timeout
from sentry.utils import jwt, safe, sdk
from sentry.utils.json import JSONData
logger = logging.getLogger(__name__)
AppConnectCredentials = namedtuple('AppConnectCredentials', ['key_id', 'key', 'issuer_id'])
REQUEST_TIMEOUT = 30.0

class RequestError(Exception):
    """An error from the response."""
    pass

class UnauthorizedError(RequestError):
    """Unauthorised: invalid, expired or revoked authentication token."""
    pass

class ForbiddenError(RequestError):
    """Forbidden: authentication token does not have sufficient permissions."""
    pass

class NoDsymUrl(enum.Enum):
    """Indicates the reason of absence of a dSYM URL from :class:`BuildInfo`."""
    PENDING = enum.auto()
    NOT_NEEDED = enum.auto()

@dataclasses.dataclass(frozen=True)
class BuildInfo:
    """Information about an App Store Connect build.

    A build is identified by the tuple of (app_id, platform, version, build_number), though
    Apple mostly names these differently.
    """
    app_id: str
    platform: str
    version: str
    build_number: str
    uploaded_date: datetime.datetime
    dsym_url: Union[NoDsymUrl, str]

def _get_authorization_header(credentials: AppConnectCredentials, expiry_sec: Optional[int]=None) -> Mapping[str, str]:
    if False:
        i = 10
        return i + 15
    'Creates a JWT (javascript web token) for use with app store connect API\n\n    All requests to app store connect require an "Authorization" header build as below.\n\n    Note: The maximum allowed expiry time is 20 minutes.  The default is somewhat shorter\n    than that to avoid running into the limit.\n\n    :return: the Bearer auth token to be added as the  "Authorization" header\n    '
    if expiry_sec is None:
        expiry_sec = 60 * 10
    with sentry_sdk.start_span(op='jwt', description='Generating AppStoreConnect JWT token'):
        token = jwt.encode({'iss': credentials.issuer_id, 'exp': int(time.time()) + expiry_sec, 'aud': 'appstoreconnect-v1'}, credentials.key, algorithm='ES256', headers={'kid': credentials.key_id, 'alg': 'ES256', 'typ': 'JWT'})
        return jwt.authorization_header(token)

def _get_appstore_json(session: Session, credentials: AppConnectCredentials, url: str) -> Mapping[str, Any]:
    if False:
        return 10
    'Returns response data from an appstore URL.\n\n    It builds and makes the request and extracts the data from the response.\n\n    :returns: a dictionary with the requested data or None if the call fails.\n\n    :raises ValueError: if the request failed or the response body could not be parsed as\n       JSON.\n    '
    with sentry_sdk.start_span(op='appconnect-request', description='AppStoreConnect API request'):
        headers = _get_authorization_header(credentials)
        if not url.startswith('https://'):
            full_url = 'https://api.appstoreconnect.apple.com'
            if url[0] != '/':
                full_url += '/'
        else:
            full_url = ''
        full_url += url
        logger.debug(f'GET {full_url}')
        with sentry_sdk.start_transaction(op='http', description='AppStoreConnect request'):
            response = session.get(full_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if not response.ok:
            err_info = {'url': full_url, 'status_code': response.status_code}
            try:
                err_info['json'] = response.json()
            except Exception:
                err_info['text'] = response.text
            with sentry_sdk.configure_scope() as scope:
                scope.set_extra('http.appconnect.api', err_info)
            if response.status_code == HTTPStatus.UNAUTHORIZED:
                raise UnauthorizedError(full_url)
            elif response.status_code == HTTPStatus.FORBIDDEN:
                raise ForbiddenError(full_url)
            else:
                raise RequestError(full_url)
        try:
            return response.json()
        except Exception as e:
            raise ValueError('Response body not JSON', full_url, response.status_code, response.text) from e

def _get_next_page(response_json: Mapping[str, Any]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Gets the URL for the next page from an App Store Connect paged response.'
    return safe.get_path(response_json, 'links', 'next')

def _get_appstore_info_paged(session: Session, credentials: AppConnectCredentials, url: str) -> Generator[JSONData, None, None]:
    if False:
        while True:
            i = 10
    'Iterates through all the pages from a paged response.\n\n    App Store Connect responses shares the general format:\n\n    data:\n      - list of elements\n    included:\n      - list of included relations as requested\n    links:\n      next: link to the next page\n    ...\n\n    The function iterates through all pages (following the next link) until\n    there is no next page, and returns a generator containing all pages\n\n    :return: a generator with the pages.\n    '
    next_url: Optional[str] = url
    while next_url is not None:
        response = _get_appstore_json(session, credentials, next_url)
        yield response
        next_url = _get_next_page(response)
_RelType = NewType('_RelType', str)
_RelId = NewType('_RelId', str)

class _IncludedRelations:
    """Related data which was returned with a page.

    The API allows to add an ``&include=some,types`` query parameter to the URLs which will
    automatically include related data of those types which are referred in the data of the
    page to be returned in the same request.  This class extracts this information from the
    page and makes it available to look up.

    :param data: The entire page data, the constructor will extract the included relations
       from this.
    """

    def __init__(self, page_data: JSONData):
        if False:
            for i in range(10):
                print('nop')
        self._items: Dict[Tuple[_RelType, _RelId], JSONData] = {}
        for relation in page_data.get('included', []):
            rel_type = _RelType(relation['type'])
            rel_id = _RelId(relation['id'])
            self._items[rel_type, rel_id] = relation

    def get_related(self, data: JSONData, relation: str) -> Optional[JSONData]:
        if False:
            while True:
                i = 10
        "Returns the named relation of the object.\n\n        ``data`` must be a JSON object which has a ``relationships`` object and\n        ``relation`` is the key of the specific related data in this list required.  This\n        function will read the object type and id from the relationships and look up the\n        actual object in the page's related data.\n        "
        rel_ptr_data = safe.get_path(data, 'relationships', relation, 'data')
        if rel_ptr_data is None:
            return None
        assert isinstance(rel_ptr_data, dict)
        rel_type = _RelType(rel_ptr_data['type'])
        rel_id = _RelId(rel_ptr_data['id'])
        return self._items[rel_type, rel_id]

    def get_multiple_related(self, data: JSONData, relation: str) -> Optional[List[JSONData]]:
        if False:
            while True:
                i = 10
        'Returns a list of all the related objects of the named relation type.\n\n        This is like :meth:`get_related` but is for relation types which have a list of\n        related objects instead of exactly one.  An example of this is a ``build`` can have\n        multiple ``buildBundles`` related to it.\n\n        Having this as a separate method makes it easier to handle the type checking.\n        '
        rel_ptr_data = safe.get_path(data, 'relationships', relation, 'data')
        if rel_ptr_data is None:
            return None
        assert isinstance(rel_ptr_data, list)
        all_related = []
        for relationship in rel_ptr_data:
            rel_type = _RelType(relationship['type'])
            rel_id = _RelId(relationship['id'])
            related_item = self._items[rel_type, rel_id]
            if related_item:
                all_related.append(related_item)
        return all_related

def get_build_info(session: Session, credentials: AppConnectCredentials, app_id: str, *, include_expired: bool=False) -> List[BuildInfo]:
    if False:
        i = 10
        return i + 15
    'Returns the build infos for an application.\n\n    The release build version information has the following structure:\n    platform: str - the platform for the build (e.g. IOS, MAC_OS ...)\n    version: str - the short version build info ( e.g. \'1.0.1\'), also called "train"\n       in starship documentation\n    build_number: str - the version of the build (e.g. \'101\'), looks like the build number\n    uploaded_date: datetime - when the build was uploaded to App Store Connect\n    '
    with sentry_sdk.start_span(op='appconnect-list-builds', description='List all AppStoreConnect builds'):
        url = f'v1/builds?filter[app]={app_id}&limit=200&include=preReleaseVersion,appStoreVersion,buildBundles&limit[buildBundles]=50&sort=-uploadedDate&filter[processingState]=VALID'
        if not include_expired:
            url += '&filter[expired]=false'
        pages = _get_appstore_info_paged(session, credentials, url)
        build_info = []
        for page in pages:
            relations = _IncludedRelations(page)
            for build in page['data']:
                try:
                    related_appstore_version = relations.get_related(build, 'appStoreVersion')
                    related_prerelease_version = relations.get_related(build, 'preReleaseVersion')
                    if related_prerelease_version:
                        version = related_prerelease_version['attributes']['version']
                        platform = related_prerelease_version['attributes']['platform']
                    elif related_appstore_version:
                        version = related_appstore_version['attributes']['versionString']
                        platform = related_appstore_version['attributes']['platform']
                    else:
                        raise KeyError('missing related version')
                    build_number = build['attributes']['version']
                    uploaded_date = parse_date(build['attributes']['uploadedDate'])
                    build_bundles = relations.get_multiple_related(build, 'buildBundles')
                    with sentry_sdk.push_scope() as scope:
                        scope.set_context('App Store Connect Build', {'build': build, 'build_bundles': build_bundles})
                        dsym_url = _get_dsym_url(build_bundles)
                    build_info.append(BuildInfo(app_id=app_id, platform=platform, version=version, build_number=build_number, uploaded_date=uploaded_date, dsym_url=dsym_url))
                except Exception:
                    logger.error('Failed to process AppStoreConnect build from API: %s', build, exc_info=True)
        return build_info

def _get_dsym_url(bundles: Optional[List[JSONData]]) -> Union[NoDsymUrl, str]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the dSYMs URL from the extracted from the build bundles.'
    if not bundles:
        return NoDsymUrl.NOT_NEEDED
    get_bundle_url: Callable[[JSONData], Any] = lambda bundle: safe.get_path(bundle, 'attributes', 'dSYMUrl', default=NoDsymUrl.NOT_NEEDED)
    app_clip_urls = [get_bundle_url(b) for b in bundles if safe.get_path(b, 'attributes', 'bundleType', default='APP') == 'APP_CLIP']
    if not all((isinstance(url, NoDsymUrl) for url in app_clip_urls)):
        sentry_sdk.capture_message("App Clip's bundle has a dSYMUrl")
    app_bundles = [app_bundle for app_bundle in bundles if safe.get_path(app_bundle, 'attributes', 'bundleType', default='APP') != 'APP_CLIP']
    if not app_bundles:
        return NoDsymUrl.NOT_NEEDED
    elif len(app_bundles) > 1:
        sentry_sdk.capture_message('len(buildBundles) != 1')
    url = get_bundle_url(app_bundles[0])
    if isinstance(url, (NoDsymUrl, str)):
        return url
    else:
        raise ValueError(f"Unexpected value in build bundle's dSYMUrl: {url}")
AppInfo = namedtuple('AppInfo', ['name', 'bundle_id', 'app_id'])

def get_apps(session: Session, credentials: AppConnectCredentials) -> Optional[List[AppInfo]]:
    if False:
        print('Hello World!')
    '\n    Returns the available applications from an account\n    :return: a list of available applications or None if the login failed, an empty list\n    means that the login was successful but there were no applications available\n    '
    url = 'v1/apps'
    ret_val = []
    try:
        app_pages = _get_appstore_info_paged(session, credentials, url)
        for app_page in app_pages:
            for app in safe.get_path(app_page, 'data', default=[]):
                app_info = AppInfo(app_id=app.get('id'), bundle_id=safe.get_path(app, 'attributes', 'bundleId'), name=safe.get_path(app, 'attributes', 'name'))
                if app_info.app_id is not None and app_info.bundle_id is not None and (app_info.name is not None):
                    ret_val.append(app_info)
                else:
                    logger.error('Malformed AppStoreConnect `apps` data')
    except ValueError:
        return None
    return ret_val

def download_dsyms(session: Session, credentials: AppConnectCredentials, url: str, path: pathlib.Path) -> None:
    if False:
        return 10
    'Downloads dSYMs at `url` into `path` which must be a filename.'
    headers = _get_authorization_header(credentials)
    with session.get(url, headers=headers, stream=True, timeout=15) as res:
        status = res.status_code
        if status == HTTPStatus.UNAUTHORIZED:
            raise UnauthorizedError
        elif status == HTTPStatus.FORBIDDEN:
            raise ForbiddenError
        elif status != HTTPStatus.OK:
            raise RequestError(f'Bad status code downloading dSYM: {status}')
        start = time.time()
        bytes_count = 0
        with open(path, 'wb') as fp:
            for chunk in res.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE):
                if time.time() - start > 315:
                    with sdk.configure_scope() as scope:
                        scope.set_extra('dSYM.bytes_fetched', bytes_count)
                    raise Timeout('Timeout during dSYM download')
                bytes_count += len(chunk)
                fp.write(chunk)