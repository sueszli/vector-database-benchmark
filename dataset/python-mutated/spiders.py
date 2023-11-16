import json
import os
import re
from typing import Callable, Iterator, List, Optional, Union
from urllib.parse import urlparse, urlsplit
import scrapy
from scrapy.http import Request, Response
from scrapy.linkextractors import IGNORED_EXTENSIONS
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from scrapy.spidermiddlewares.httperror import HttpError
from scrapy.utils.url import url_has_any_extension
from twisted.python.failure import Failure
EXCLUDED_DOMAINS = ['github.com', 'gist.github.com', 'www.amazon.com', 'gitlab.com']
EXCLUDED_URLS = ['https://calendar.google.com/calendar/embed?src=ktiduof4eoh47lmgcl2qunnc0o@group.calendar.google.com', 'https://medium.freecodecamp.org/', 'https://www.git-tower.com/blog/command-line-cheat-sheet/', 'https://marketplace.visualstudio.com/items?itemName=rafaelmaiolla.remote-vscode', 'https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh', 'https://www.linkedin.com/company/zulip-project', 'https://giphy.com', 'https://giphy.com/apps/giphycapture', 'https://www.udemy.com/course/the-complete-react-native-and-redux-course/']
VNU_IGNORE = ['Attribute “markdown” not allowed on element “div” at this point\\.', 'No “p” element in scope but a “p” end tag seen\\.', 'Element “div” not allowed as child of element “ul” in this context\\. \\(Suppressing further errors from this subtree\\.\\)', 'Trailing slash on void elements has no effect and interacts badly with unquoted attribute values\\.']
VNU_IGNORE_REGEX = re.compile('|'.join(VNU_IGNORE))
DEPLOY_ROOT = os.path.abspath(os.path.join(__file__, '../../../../../..'))
ZULIP_SERVER_GITHUB_FILE_PATH_PREFIX = '/zulip/zulip/blob/main'
ZULIP_SERVER_GITHUB_DIRECTORY_PATH_PREFIX = '/zulip/zulip/tree/main'

class BaseDocumentationSpider(scrapy.Spider):
    name: Optional[str] = None
    deny_domains: List[str] = []
    start_urls: List[str] = []
    deny: List[str] = []
    file_extensions: List[str] = ['.' + ext for ext in IGNORED_EXTENSIONS]
    tags = ('a', 'area', 'img')
    attrs = ('href', 'src')

    def _has_extension(self, url: str) -> bool:
        if False:
            while True:
                i = 10
        return url_has_any_extension(url, self.file_extensions)

    def _is_external_url(self, url: str) -> bool:
        if False:
            i = 10
            return i + 15
        return url.startswith('http') or self._has_extension(url)

    def check_existing(self, response: Response) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.log(response)

    def _is_external_link(self, url: str) -> bool:
        if False:
            while True:
                i = 10
        split_url = urlsplit(url)
        if split_url.hostname == 'chat.zulip.org':
            return True
        if split_url.hostname == 'zulip.readthedocs.io' or f'.{split_url.hostname}'.endswith(('.zulip.com', '.zulip.org')):
            return False
        if split_url.scheme == 'file' or split_url.hostname == 'localhost':
            return False
        if split_url.hostname == 'github.com' and f'{split_url.path}/'.startswith((f'{ZULIP_SERVER_GITHUB_FILE_PATH_PREFIX}/', f'{ZULIP_SERVER_GITHUB_DIRECTORY_PATH_PREFIX}/')):
            return False
        if split_url.hostname == 'github.com' and split_url.path.startswith('/zulip/'):
            return True
        return True

    def check_fragment(self, response: Response) -> None:
        if False:
            i = 10
            return i + 15
        self.log(response)
        xpath_template = "//*[@id='{fragment}' or @name='{fragment}']"
        fragment = urlsplit(response.request.url).fragment
        if not response.selector.xpath(xpath_template.format(fragment=fragment)):
            self.logger.error('Fragment #%s is not found on page %s', fragment, response.request.url)

    def _vnu_callback(self, url: str) -> Callable[[Response], None]:
        if False:
            print('Hello World!')

        def callback(response: Response) -> None:
            if False:
                for i in range(10):
                    print('nop')
            vnu_out = json.loads(response.text)
            for message in vnu_out['messages']:
                if not VNU_IGNORE_REGEX.fullmatch(message['message']):
                    self.logger.error('"%s":%d.%d-%d.%d: %s: %s', url, message.get('firstLine', message['lastLine']), message.get('firstColumn', message['lastColumn']), message['lastLine'], message['lastColumn'], message['type'], message['message'])
        return callback

    def _make_requests(self, url: str) -> Iterator[Request]:
        if False:
            while True:
                i = 10
        split_url = urlsplit(url)
        if split_url.netloc == 'localhost:9981' and split_url.path in ['', '/']:
            return
        if url.startswith('http://localhost:9981/communities'):
            return
        callback: Callable[[Response], Optional[Iterator[Request]]] = self.parse
        dont_filter = False
        method = 'GET'
        if self._is_external_url(url):
            callback = self.check_existing
            method = 'HEAD'
            if split_url.hostname == 'github.com' and f'{split_url.path}/'.startswith(f'{ZULIP_SERVER_GITHUB_FILE_PATH_PREFIX}/'):
                file_path = DEPLOY_ROOT + split_url.path[len(ZULIP_SERVER_GITHUB_FILE_PATH_PREFIX):]
                if not os.path.isfile(file_path):
                    self.logger.error('There is no local file associated with the GitHub URL: %s', url)
                return
            elif split_url.hostname == 'github.com' and f'{split_url.path}/'.startswith(f'{ZULIP_SERVER_GITHUB_DIRECTORY_PATH_PREFIX}/'):
                dir_path = DEPLOY_ROOT + split_url.path[len(ZULIP_SERVER_GITHUB_DIRECTORY_PATH_PREFIX):]
                if not os.path.isdir(dir_path):
                    self.logger.error('There is no local directory associated with the GitHub URL: %s', url)
                return
        elif split_url.fragment != '':
            dont_filter = True
            callback = self.check_fragment
        if getattr(self, 'skip_external', False) and self._is_external_link(url):
            return
        if split_url.hostname in EXCLUDED_DOMAINS:
            return
        if url in EXCLUDED_URLS:
            return
        yield Request(url, method=method, callback=callback, dont_filter=dont_filter, errback=self.error_callback)

    def start_requests(self) -> Iterator[Request]:
        if False:
            for i in range(10):
                print('nop')
        for url in self.start_urls:
            yield from self._make_requests(url)

    def parse(self, response: Response) -> Iterator[Request]:
        if False:
            while True:
                i = 10
        self.log(response)
        if getattr(self, 'validate_html', False):
            yield Request('http://127.0.0.1:9988/?out=json', method='POST', headers={'Content-Type': response.headers['Content-Type']}, body=response.body, callback=self._vnu_callback(response.url), errback=self.error_callback)
        for link in LxmlLinkExtractor(deny_domains=self.deny_domains, deny_extensions=['doc'], tags=self.tags, attrs=self.attrs, deny=self.deny, canonicalize=False).extract_links(response):
            yield from self._make_requests(link.url)

    def retry_request_with_get(self, request: Request) -> Iterator[Request]:
        if False:
            print('Hello World!')
        request.method = 'GET'
        request.dont_filter = True
        yield request

    def error_callback(self, failure: Failure) -> Optional[Union[Failure, Iterator[Request]]]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(failure.value, HttpError):
            response = failure.value.response
            if urlparse(response.url).netloc == 'idmsa.apple.com':
                return None
            if response.status == 405 and response.request.method == 'HEAD':
                return self.retry_request_with_get(response.request)
            self.logger.error('Please check link: %s', response.request.url)
        return failure