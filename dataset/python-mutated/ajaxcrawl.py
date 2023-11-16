from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Union
from w3lib import html
from scrapy import Request, Spider
from scrapy.crawler import Crawler
from scrapy.exceptions import NotConfigured
from scrapy.http import HtmlResponse, Response
from scrapy.settings import BaseSettings
if TYPE_CHECKING:
    from typing_extensions import Self
logger = logging.getLogger(__name__)

class AjaxCrawlMiddleware:
    """
    Handle 'AJAX crawlable' pages marked as crawlable via meta tag.
    For more info see https://developers.google.com/webmasters/ajax-crawling/docs/getting-started.
    """

    def __init__(self, settings: BaseSettings):
        if False:
            print('Hello World!')
        if not settings.getbool('AJAXCRAWL_ENABLED'):
            raise NotConfigured
        self.lookup_bytes: int = settings.getint('AJAXCRAWL_MAXSIZE', 32768)

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if False:
            while True:
                i = 10
        return cls(crawler.settings)

    def process_response(self, request: Request, response: Response, spider: Spider) -> Union[Request, Response]:
        if False:
            print('Hello World!')
        if not isinstance(response, HtmlResponse) or response.status != 200:
            return response
        if request.method != 'GET':
            return response
        if 'ajax_crawlable' in request.meta:
            return response
        if not self._has_ajax_crawlable_variant(response):
            return response
        ajax_crawl_request = request.replace(url=request.url + '#!')
        logger.debug('Downloading AJAX crawlable %(ajax_crawl_request)s instead of %(request)s', {'ajax_crawl_request': ajax_crawl_request, 'request': request}, extra={'spider': spider})
        ajax_crawl_request.meta['ajax_crawlable'] = True
        return ajax_crawl_request

    def _has_ajax_crawlable_variant(self, response: Response) -> bool:
        if False:
            while True:
                i = 10
        '\n        Return True if a page without hash fragment could be "AJAX crawlable"\n        according to https://developers.google.com/webmasters/ajax-crawling/docs/getting-started.\n        '
        body = response.text[:self.lookup_bytes]
        return _has_ajaxcrawlable_meta(body)
_ajax_crawlable_re: re.Pattern[str] = re.compile('<meta\\s+name=["\\\']fragment["\\\']\\s+content=["\\\']!["\\\']/?>')

def _has_ajaxcrawlable_meta(text: str) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    >>> _has_ajaxcrawlable_meta(\'<html><head><meta name="fragment"  content="!"/></head><body></body></html>\')\n    True\n    >>> _has_ajaxcrawlable_meta("<html><head><meta name=\'fragment\' content=\'!\'></head></html>")\n    True\n    >>> _has_ajaxcrawlable_meta(\'<html><head><!--<meta name="fragment"  content="!"/>--></head><body></body></html>\')\n    False\n    >>> _has_ajaxcrawlable_meta(\'<html></html>\')\n    False\n    '
    if 'fragment' not in text:
        return False
    if 'content' not in text:
        return False
    text = html.remove_tags_with_content(text, ('script', 'noscript'))
    text = html.replace_entities(text)
    text = html.remove_comments(text)
    return _ajax_crawlable_re.search(text) is not None