from __future__ import annotations
from email.utils import formatdate
from typing import TYPE_CHECKING, Optional, Union
from twisted.internet import defer
from twisted.internet.error import ConnectError, ConnectionDone, ConnectionLost, ConnectionRefusedError, DNSLookupError, TCPTimedOutError, TimeoutError
from twisted.web.client import ResponseFailed
from scrapy import signals
from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http.request import Request
from scrapy.http.response import Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.misc import load_object
if TYPE_CHECKING:
    from typing_extensions import Self

class HttpCacheMiddleware:
    DOWNLOAD_EXCEPTIONS = (defer.TimeoutError, TimeoutError, DNSLookupError, ConnectionRefusedError, ConnectionDone, ConnectError, ConnectionLost, TCPTimedOutError, ResponseFailed, OSError)

    def __init__(self, settings: Settings, stats: StatsCollector) -> None:
        if False:
            print('Hello World!')
        if not settings.getbool('HTTPCACHE_ENABLED'):
            raise NotConfigured
        self.policy = load_object(settings['HTTPCACHE_POLICY'])(settings)
        self.storage = load_object(settings['HTTPCACHE_STORAGE'])(settings)
        self.ignore_missing = settings.getbool('HTTPCACHE_IGNORE_MISSING')
        self.stats = stats

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if False:
            print('Hello World!')
        assert crawler.stats
        o = cls(crawler.settings, crawler.stats)
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(o.spider_closed, signal=signals.spider_closed)
        return o

    def spider_opened(self, spider: Spider) -> None:
        if False:
            while True:
                i = 10
        self.storage.open_spider(spider)

    def spider_closed(self, spider: Spider) -> None:
        if False:
            return 10
        self.storage.close_spider(spider)

    def process_request(self, request: Request, spider: Spider) -> Union[Request, Response, None]:
        if False:
            return 10
        if request.meta.get('dont_cache', False):
            return None
        if not self.policy.should_cache_request(request):
            request.meta['_dont_cache'] = True
            return None
        cachedresponse: Optional[Response] = self.storage.retrieve_response(spider, request)
        if cachedresponse is None:
            self.stats.inc_value('httpcache/miss', spider=spider)
            if self.ignore_missing:
                self.stats.inc_value('httpcache/ignore', spider=spider)
                raise IgnoreRequest(f'Ignored request not in cache: {request}')
            return None
        cachedresponse.flags.append('cached')
        if self.policy.is_cached_response_fresh(cachedresponse, request):
            self.stats.inc_value('httpcache/hit', spider=spider)
            return cachedresponse
        request.meta['cached_response'] = cachedresponse
        return None

    def process_response(self, request: Request, response: Response, spider: Spider) -> Union[Request, Response]:
        if False:
            return 10
        if request.meta.get('dont_cache', False):
            return response
        if 'cached' in response.flags or '_dont_cache' in request.meta:
            request.meta.pop('_dont_cache', None)
            return response
        if 'Date' not in response.headers:
            response.headers['Date'] = formatdate(usegmt=True)
        cachedresponse: Optional[Response] = request.meta.pop('cached_response', None)
        if cachedresponse is None:
            self.stats.inc_value('httpcache/firsthand', spider=spider)
            self._cache_response(spider, response, request, cachedresponse)
            return response
        if self.policy.is_cached_response_valid(cachedresponse, response, request):
            self.stats.inc_value('httpcache/revalidate', spider=spider)
            return cachedresponse
        self.stats.inc_value('httpcache/invalidate', spider=spider)
        self._cache_response(spider, response, request, cachedresponse)
        return response

    def process_exception(self, request: Request, exception: Exception, spider: Spider) -> Union[Request, Response, None]:
        if False:
            while True:
                i = 10
        cachedresponse: Optional[Response] = request.meta.pop('cached_response', None)
        if cachedresponse is not None and isinstance(exception, self.DOWNLOAD_EXCEPTIONS):
            self.stats.inc_value('httpcache/errorrecovery', spider=spider)
            return cachedresponse
        return None

    def _cache_response(self, spider: Spider, response: Response, request: Request, cachedresponse: Optional[Response]) -> None:
        if False:
            i = 10
            return i + 15
        if self.policy.should_cache_response(response, request):
            self.stats.inc_value('httpcache/store', spider=spider)
            self.storage.store_response(spider, request, response)
        else:
            self.stats.inc_value('httpcache/uncacheable', spider=spider)