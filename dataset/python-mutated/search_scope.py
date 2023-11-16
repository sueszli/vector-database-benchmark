import itertools
import logging
import os
import posixpath
import urllib.parse
from typing import List
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.models.index import PyPI
from pip._internal.utils.compat import has_tls
from pip._internal.utils.misc import normalize_path, redact_auth_from_url
logger = logging.getLogger(__name__)

class SearchScope:
    """
    Encapsulates the locations that pip is configured to search.
    """
    __slots__ = ['find_links', 'index_urls', 'no_index']

    @classmethod
    def create(cls, find_links: List[str], index_urls: List[str], no_index: bool) -> 'SearchScope':
        if False:
            return 10
        '\n        Create a SearchScope object after normalizing the `find_links`.\n        '
        built_find_links: List[str] = []
        for link in find_links:
            if link.startswith('~'):
                new_link = normalize_path(link)
                if os.path.exists(new_link):
                    link = new_link
            built_find_links.append(link)
        if not has_tls():
            for link in itertools.chain(index_urls, built_find_links):
                parsed = urllib.parse.urlparse(link)
                if parsed.scheme == 'https':
                    logger.warning('pip is configured with locations that require TLS/SSL, however the ssl module in Python is not available.')
                    break
        return cls(find_links=built_find_links, index_urls=index_urls, no_index=no_index)

    def __init__(self, find_links: List[str], index_urls: List[str], no_index: bool) -> None:
        if False:
            i = 10
            return i + 15
        self.find_links = find_links
        self.index_urls = index_urls
        self.no_index = no_index

    def get_formatted_locations(self) -> str:
        if False:
            i = 10
            return i + 15
        lines = []
        redacted_index_urls = []
        if self.index_urls and self.index_urls != [PyPI.simple_url]:
            for url in self.index_urls:
                redacted_index_url = redact_auth_from_url(url)
                purl = urllib.parse.urlsplit(redacted_index_url)
                if not purl.scheme and (not purl.netloc):
                    logger.warning('The index url "%s" seems invalid, please provide a scheme.', redacted_index_url)
                redacted_index_urls.append(redacted_index_url)
            lines.append('Looking in indexes: {}'.format(', '.join(redacted_index_urls)))
        if self.find_links:
            lines.append('Looking in links: {}'.format(', '.join((redact_auth_from_url(url) for url in self.find_links))))
        return '\n'.join(lines)

    def get_index_urls_locations(self, project_name: str) -> List[str]:
        if False:
            return 10
        'Returns the locations found via self.index_urls\n\n        Checks the url_name on the main (first in the list) index and\n        use this url_name to produce all locations\n        '

        def mkurl_pypi_url(url: str) -> str:
            if False:
                for i in range(10):
                    print('nop')
            loc = posixpath.join(url, urllib.parse.quote(canonicalize_name(project_name)))
            if not loc.endswith('/'):
                loc = loc + '/'
            return loc
        return [mkurl_pypi_url(url) for url in self.index_urls]