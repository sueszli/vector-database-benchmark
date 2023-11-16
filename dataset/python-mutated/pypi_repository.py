from __future__ import annotations
import logging
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Any
import requests
import requests.adapters
from cachecontrol.controller import logger as cache_control_logger
from poetry.core.packages.package import Package
from poetry.core.packages.utils.link import Link
from poetry.core.version.exceptions import InvalidVersion
from poetry.repositories.exceptions import PackageNotFound
from poetry.repositories.http_repository import HTTPRepository
from poetry.repositories.link_sources.json import SimpleJsonPage
from poetry.repositories.parsers.pypi_search_parser import SearchResultParser
from poetry.utils.constants import REQUESTS_TIMEOUT
cache_control_logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from packaging.utils import NormalizedName
    from poetry.core.constraints.version import Version
    from poetry.core.constraints.version import VersionConstraint
SUPPORTED_PACKAGE_TYPES = {'sdist', 'bdist_wheel'}

class PyPiRepository(HTTPRepository):

    def __init__(self, url: str='https://pypi.org/', disable_cache: bool=False, fallback: bool=True, pool_size: int=requests.adapters.DEFAULT_POOLSIZE) -> None:
        if False:
            while True:
                i = 10
        super().__init__('PyPI', url.rstrip('/') + '/simple/', disable_cache=disable_cache, pool_size=pool_size)
        self._base_url = url
        self._fallback = fallback

    def search(self, query: str) -> list[Package]:
        if False:
            for i in range(10):
                print('nop')
        results = []
        response = requests.get(self._base_url + 'search', params={'q': query}, timeout=REQUESTS_TIMEOUT)
        parser = SearchResultParser()
        parser.feed(response.text)
        for result in parser.results:
            try:
                package = Package(result.name, result.version)
                package.description = result.description.strip()
                results.append(package)
            except InvalidVersion:
                self._log(f'Unable to parse version "{result.version}" for the {result.name} package, skipping', level='debug')
        return results

    def get_package_info(self, name: NormalizedName) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Return the package information given its name.\n\n        The information is returned from the cache if it exists\n        or retrieved from the remote server.\n        '
        return self._get_package_info(name)

    def _find_packages(self, name: NormalizedName, constraint: VersionConstraint) -> list[Package]:
        if False:
            while True:
                i = 10
        '\n        Find packages on the remote server.\n        '
        try:
            json_page = self.get_page(name)
        except PackageNotFound:
            self._log(f'No packages found for {name}', level='debug')
            return []
        versions = [(version, json_page.yanked(name, version)) for version in json_page.versions(name) if constraint.allows(version)]
        return [Package(name, version, yanked=yanked) for (version, yanked) in versions]

    def _get_package_info(self, name: NormalizedName) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        headers = {'Accept': 'application/vnd.pypi.simple.v1+json'}
        info = self._get(f'simple/{name}/', headers=headers)
        if info is None:
            raise PackageNotFound(f'Package [{name}] not found.')
        return info

    def find_links_for_package(self, package: Package) -> list[Link]:
        if False:
            while True:
                i = 10
        json_data = self._get(f'pypi/{package.name}/{package.version}/json')
        if json_data is None:
            return []
        links = []
        for url in json_data['urls']:
            if url['packagetype'] in SUPPORTED_PACKAGE_TYPES:
                h = f"sha256={url['digests']['sha256']}"
                links.append(Link(url['url'] + '#' + h, yanked=self._get_yanked(url)))
        return links

    def _get_release_info(self, name: NormalizedName, version: Version) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        from poetry.inspection.info import PackageInfo
        self._log(f'Getting info for {name} ({version}) from PyPI', 'debug')
        json_data = self._get(f'pypi/{name}/{version}/json')
        if json_data is None:
            raise PackageNotFound(f'Package [{name}] not found.')
        info = json_data['info']
        data = PackageInfo(name=info['name'], version=info['version'], summary=info['summary'], requires_dist=info['requires_dist'], requires_python=info['requires_python'], files=info.get('files', []), yanked=self._get_yanked(info), cache_version=str(self.CACHE_VERSION))
        try:
            version_info = json_data['urls']
        except KeyError:
            version_info = []
        for file_info in version_info:
            if file_info['packagetype'] in SUPPORTED_PACKAGE_TYPES:
                data.files.append({'file': file_info['filename'], 'hash': 'sha256:' + file_info['digests']['sha256']})
        if self._fallback and data.requires_dist is None:
            self._log('No dependencies found, downloading archives', level='debug')
            urls = defaultdict(list)
            for url in json_data['urls']:
                dist_type = url['packagetype']
                if dist_type not in SUPPORTED_PACKAGE_TYPES:
                    continue
                urls[dist_type].append(url['url'])
            if not urls:
                return data.asdict()
            info = self._get_info_from_urls(urls)
            data.requires_dist = info.requires_dist
            if not data.requires_python:
                data.requires_python = info.requires_python
        return data.asdict()

    def _get_page(self, name: NormalizedName) -> SimpleJsonPage:
        if False:
            for i in range(10):
                print('nop')
        source = self._base_url + f'simple/{name}/'
        info = self.get_package_info(name)
        return SimpleJsonPage(source, info)

    def _get(self, endpoint: str, headers: dict[str, str] | None=None) -> dict[str, Any] | None:
        if False:
            i = 10
            return i + 15
        try:
            json_response = self.session.get(self._base_url + endpoint, raise_for_status=False, timeout=REQUESTS_TIMEOUT, headers=headers)
        except requests.exceptions.TooManyRedirects:
            self.session.delete_cache(self._base_url + endpoint)
            json_response = self.session.get(self._base_url + endpoint, raise_for_status=False, timeout=REQUESTS_TIMEOUT, headers=headers)
        if json_response.status_code != 200:
            return None
        json: dict[str, Any] = json_response.json()
        return json

    @staticmethod
    def _get_yanked(json_data: dict[str, Any]) -> str | bool:
        if False:
            return 10
        if json_data.get('yanked', False):
            return json_data.get('yanked_reason') or True
        return False